<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22 晚最终定稿）：需修复 —— 采用算子组交付的新算子，本条提交、点名修复
     决定：适配组确认采用 dlc_kda_causal_conv_prefill，由算子组修复「N 分块」卡死 + 「direct_state 忽略
     state stride（真实池子上寻址错误）」两个 P0，以及 host 校验下沉 validate_data、被跳过行输出预置零等
     附带项后接入；修复期间插件用 torch._causal_conv1d 兜底。
     完整修复单（可直接粘贴）见 给算子组_01_02_修复要点.md -->
<!-- 2026-09-22 新算子实测：小 N 数值正确（out_err 2.4e-04、state_err=0），但 N>=3072 卡死设备
     （整条序列驻留 16MB VMEM、无 token 分块）。
     2026-09-22 晚复核修正：初版写的「host 只收 1D state_indices」「host 只收 fp32 weight」两条**已作废** ——
     GLM5 prefill 上游本来就是 1D cache_indices（gdn_attn.py:257/330 取 block_table[:,0]；上游契约
     causal_conv1d.py:524-525「cache_indices: (batch)」），且 _merged_conv_weight 本来就是 fp32 [D,4]
     （kda.py:211-215 params_dtype=torch.float32、:396-405 拼成 [3D,4]）。两条均**无需修改**。 -->
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：已被 torch._causal_conv1d 覆盖（建议撤回本新增需求）
     证据：KDA 形状 prefill 实测 out_err=1.95e-03（bf16 级），conv_state 逐元素一致；constraint：x 需 [D,T] channel 连续、weight 需 2D [D,K]、conv_states 需通道维 stride==1、bf16 storage_offset 需为偶数、APC 参数必须为 None
     建议：撤回本条，改由插件把 GLM5 KDA 接到 torch._causal_conv1d；算子组只需确认上述布局/参数约束是否长期有效
     （**该建议已被 2026-09-22 晚决定取代：本条改为提交并采用新算子，见文件顶部注释**） -->

# [Operator Request] dlc_kda_causal_conv_prefill（KDA 短卷积 prefill）

## 需求类型
修复算子功能或精度

**需求分类：需修复（本条提交，点名修复）**
- **2026-09-22 晚最终决定：采用算子组交付的新算子**，由算子组修复下述缺陷后接入；
  修复完成前插件暂用现有算子兜底（迁移开关 `KDA_CONV_USE_NEW_OP` 计划加在 `vllm_cl/patch/patch_kda.py`，默认 False，**目前尚未落到代码里**，插件现在仍走 native 组合实现）。
  完整、可直接粘贴的修复单见 [`给算子组_01_02_修复要点.md`](给算子组_01_02_修复要点.md)。

- 现有算子：`torch._causal_conv1d(x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state, block_idx_first_scheduled_token=None, block_idx_last_scheduled_token=None, initial_state_idx=None, num_computed_tokens=None, block_size_to_align=0, has_bias=True, silu_activation=True)`（实现：`aten/src/ATen/native/dlc/CausalConv1d.cxx`）。
- 实测（dlc:0；dim=3072 / K=4 / T=16 / 变长；bf16）：`out_err=1.953e-03`、`conv_state` 与 reference 逐元素一致 → KDA 短卷积 prefill 已被覆盖。
- 剩余工作全在插件侧布局适配：
  - `weight` 需 2D `[dim, K]`（KDA 的 `[dim, 1, K]` 需 `squeeze(1)`）；
  - `x` 需 `[dim, T]` 且通道维连续（channel-last）；
  - `conv_states` 需通道维 `stride == 1`（由 `[pages, tap, dim]` buffer `permute(0, 2, 1)` 得到），否则报 `conv state dim axis must be contiguous`；
  - bf16 下 `storage_offset` 需为偶数；APC 参数（`block_idx_*` / `initial_state_idx` / `num_computed_tokens`）目前必须为 `None`。
- 上述现有算子事实**保留作为「兜底方案」**：新算子修复期间插件切到 `torch._causal_conv1d`，
  零拷贝、按 `cache_indices` 只读写活跃页、N 不受限。
- 本次调整（2026-09-22 晚）：由「可复用（撤回）」→「**需修复（提交）**」；优先级 P2→**P0**；
  算子名称改回修复 `dlc_kda_causal_conv_prefill`。理由：新算子在真机上比现有算子更省（少若干小 kernel、
  能按页读写），价值成立，但 4 个缺陷必须先修（详见 `给算子组_01_02_修复要点.md` 第一节）。


## 2026-09-22 新算子实测（算子组已交付 `dlc_kda_causal_conv_prefill`）

算子组已在 pytorch `test-glm53flash` 分支提交专用算子 `dlc_kda_causal_conv_prefill`
（`aten/src/ATen/native/dlc/CausalConv1d.cxx`，kernel `custom_kda_causal_conv_prefill`），
故对「是否还需要新算子」重新实测（脚本：`probes/probe_new_kda_conv_prefill.py`）。

**实测结论：数值正确，但长 prefill 会卡死设备，不能用于 GLM5。**

| 用例（D=3072, K=4, state=[S,D,3]） | 结果 |
|---|---|
| B=3 变长 [16,5,33] | `out_err=2.44e-04`（bf16 级）、`state_err=0.00e+00` |
| B=8 变长 [128,1,512,7,256,33,1024,3]（总 N=1964，最长段 1024） | `out_err=1.95e-03`（4 ulp bf16）、`state_err=0` |
| 短序列（<state_len）、`has_initial_state=False` | `out_err=4.88e-04`（1 ulp）、`state_err=0` |
| null slot 0 | 输出全 0、state 不变（与现有算子一致） |
| `state_len=6`（num_spec=3 的宽度） | `out_err=0`、`state_err=0` |
| 异常路径：`idx=-1` / 重复槽位 / `idx≥slots` | 三者均被 host `TORCH_CHECK` 拒绝（`RuntimeError`），**其中 `idx=-1` 被拒与现有 `_causal_conv1d` 不同** |
| D=320（非 256 倍数，host pad 路径） | `out_err=0`、`state_err=0`（pad 路径正确） |
| **N=1536 / 2048 / 2560（单请求）** | 通过（`t≈0.016ms`） |
| **N=3072 / 4096（GLM5 真实 prefill 长度）** | **卡死设备**（dlc:0、dlc:1、dlc:3 均复现；进程 200% CPU 空转，超时无法退出，必须复位设备） |

**根因（已定位到代码）**：kernel 把「整条序列」一次搬进片上 VMEM ——
`kda_causal_conv_prefill.c` 里 `vmem_input = vmem_addr`、`vmem_output = vmem_weight + weight_blocks`，
两块各按 `max_len * bf16_row_blocks` 预留，而 `kVmemSize = 0x1000000`（16 MB，
`DLCSynapse/external_includes/Veloce_Architecture.h:40`）。D=3072 时每 token 占 6144 B，
输入+输出各一份 → 约 2048~3072 token 就是上限，N=4096 直接越界写爆 VMEM。
host wrapper 也没有任何 N 上限校验或分块处理。

**补充（2026-09-22 晚，逐例复测）**：该脚本已支持逐例执行（`python probe_new_kda_conv_prefill.py <dev> 3,4,5,...`），
第 3~9 例在干净设备上全部复测通过（结果见上表），只有第 2 例（N=4096，长序列）会卡死——已确认与前面的
N=3072 是同一根因，**不要为了测边界反复重跑该长度**（每跑一次浪费一张卡）。

**处置（2026-09-22 晚最终定稿）**：
- 本条改为「**提交**」：采用新算子，请算子组修下面的问题；
- 在修复完成前，插件用 `torch._causal_conv1d` 兜底（迁移开关默认关），修好后切新算子并重跑数值对齐。
- **算子组待修事项（按优先级；完整可粘贴版见 [`给算子组_01_02_修复要点.md`](给算子组_01_02_修复要点.md) 第一节）**：
  1. **[P0] N 分块**：`dlc_kda_causal_conv_prefill` 需按 token 分块（或至少 host 侧对 `N` 加上限校验并报错），
     否则任何长 prefill（GLM5 常态 4096）都会卡死设备（实测 N=3072/4096 在 dlc:0/1/3 上三次复现，须复位才能恢复）；
  2. **[P0] `direct_state` 快路径忽略 state stride**：`make_flat_storage_view` + `params` 的设计本意是零拷贝
     （内核按 stride 逐行取 conv 段、跳过行间的 SSM 段 / 页填充），但 `kda_causal_conv_prefill.c` 只用了
     `params[0]`(state_len) 与 `params[5]`(state_slots)，地址写死成 `slot * state_len * dim_bf16 / 32`
     （`:128,178,181,185`），`params[1..4]`（dim/tap/slot stride、storage_offset）没被使用。
     GLM5 的 conv state 是 `MambaBase.bind_kv_cache` 切出来的分页视图，`stride(0) = page_size_bytes/itemsize`
     （TP8 下 ≈271,360 个元素），而 `L*D = 9,216`，相差约 29 倍，且 `direct_state` 条件照样满足
     → 读错行，并把 conv state 写进行间的 recurrent 段（静默破坏递推状态）。
     实测（`probes/probe_new_kda_conv_prefill_state_stride.py`，dlc:0 / dlc:1 一致）：紧凑行距
     `out_err=9.77e-04 / state_err=0`；池子行距 `L*D+2048` 时 `out_err=3.545 / state_err=3.453`，
     且**行间段 4096 个哨兵元素被改写**（写坏 recurrent 内存）；同一布局下现有 `torch._causal_conv1d`
     完全正确（`out_err=3.91e-03 / state_err=0`，哨兵零改写）；
     修法见 [`给算子组_01_02_修复要点.md`](给算子组_01_02_修复要点.md) 第一节「另一个 P0」；
  3. **[P2] host 校验下沉到 `validate_data` 开关（默认关，debug 时才同步）**：`CausalConv1d.cxx:497-512`
     现在**无条件** `qsl.to(kCPU)` + `indices.to(kCPU)` 再逐请求校验（单调性 / 页号范围 / 活跃槽位不重复）；
     GLM5 是 34 层 × 每次调用 = 每步 68 次强制 D2H，且无法进 CUDA graph。请照上游 `validate_data=False`
     的约定（`ops/causal_conv1d.py:498,615`）把这段放进开关分支；
  4. **[P2] 被跳过行的输出预置零**：校验默认关掉后，`-1` / 越界会直接进 kernel，而 kernel 只
     `continue` 不写输出（`kda_causal_conv_prefill.c:115`）、输出缓冲是 `at::empty`（`CausalConv1d.cxx:593`）
     → 会漏脏值。请改成 `at::zeros` 预置（现有算子 `CausalConv1d.cxx:146` 就是这么做的）；
  5. **[结论：非缺陷、不阻塞] 插件/GLM5 路径不产生负页号**：`non_spec_state_indices_tensor` 取自
     block table 首列（`gdn_attn.py:257/330`），padding 行两侧都填 `NULL_BLOCK_ID=0`
     （`vllm_cl/worker/dlc_model_runner.py:2229-2231`、`gdn_attn.py:487`），真实请求块号 ≥1。
     上游契约虽允许 `-1`（`causal_conv1d.py:534-540`），但只要上面 3+4 落实，`-1` 与越界都会退化成
     「静默跳过 + 输出全 0」，与现有 `_causal_conv1d` 一致，无需插件侧额外过滤；
  6. **[P2/可选] 与 #02 对齐**：若愿意，可让 `state_indices` 也接受 2D `[B,T]`（取第 0 列）。

- **两条已作废的旧结论（2026-09-22 晚核对源码后修正，勿再作为缺陷提交）**：
  - `state_indices` 必须 1D `[B]` **不是缺陷**：GLM5 prefill 上游本来就是 1D（见上方文件头注释）；
  - `weight` 必须 fp32 `[D,4]` **不是缺陷**：GLM5 的 `_merged_conv_weight` 本来就是 fp32 `[D,4]`。
- 复现脚本：`probes/probe_new_kda_conv_prefill.py`（**第 2 例 N=4096 会卡死设备，默认不跑**，
  必须显式写 case id 才执行；可用 `probes/probe_device_health.py` 判定/复查设备状态）。

## 2026-09-28 修复后复测（结论：两个 P0 已修复，可直接接入）

算子组在 `DLC_Custom_Kernel` `jinxing_fix_7949_260924`（commit `2049ebdde`，kernel 改为按 VMEM 分块）
+ `pytorch` `jinxing_kda_causal`（commit `0fbb10e0117`，删除 host `direct_state`/pad 路径）修复；
已 cherry-pick 到本地 `test-glm53flash`（`DLC_Custom_Kernel f266b18a5`、`pytorch 9726fb37e5f`）。
复测脚本按新 ABI 更新：`probes/probe_new_kda_conv_prefill.py`、`probes/probe_new_kda_conv_prefill_state_stride.py`。

新 ABI（以算子组交付为准，与本文件「期望 API」一节不同）：
- `conv_state` 逻辑形状 **`[S_slot, state_len, D]`**（SD，D 连续），kernel 直接 in-place 写回；
- `weight` 为**已转置**的 `[4, D]` fp32；
- `has_initial_state` 为 int32（不再接受 bool）；
- 返回单个 `out`。

| 复测项 | 修复前 | 修复后（dlc:0~7 干净卡逐例） |
|---|---|---|
| N=4096 单序列（真机 dump 几何） | **卡死设备** | 通过：`out_err=3.91e-03`、`state_err=0`、t≈1 ms |
| N=8192 / 16384 / 32768 | 未测（预期卡死） | 通过：`out_err=3.91e-03`、`state_err=0`、t≤14 ms |
| B=16、N=8192 变长批 | — | 通过：`out_err=3.91e-03`、`state_err=0` |
| 非连续 state 视图（池子块间行距 `L*D+2048`） | 忽略 stride，读错行 + 写坏行间 recurrent 段 | 通过：`out_err=0`、`state_err=0`、**行间段被改写 0 元素** |
| 短序列 / `has_initial_state=False` / null slot 0 / `state_len=6` / D=320 | — | 全部通过（`state_err=0`） |
| 异常路径（重复槽位 / `idx=-1` / `idx>=slots`） | host `TORCH_CHECK` 拒绝 | 改为 **kernel 内断言**（`duplicate non-null state slot` / `invalid state slot`）；断言会 poison 设备上下文，需单独占一张卡 |

根因与修法：kernel 改为「weight/state 常驻 VMEM + 按 VMEM 容量切 `chunk_len` 个 token」，
消除整条序列驻留 16 MB VMEM 导致的卡死；host wrapper 删除了会走零拷贝的 `direct_state` 快路径
（`CausalConv1d.cxx` 旧 `:549-551`），改为把 `conv_state` 直接作为 `kernel.output` 提交。
DLC 的 `dlc_dlc_contiguous` 提交路径会对非连续张量自动做「连续副本 + 结束后 `copy_` 回写」，
因此真实分页池上的非连续 `conv_state` 视图结果正确（代价是单 slot conv 段 18 KB 量级的一次拷贝，
远小于修复前的整池 4 遍拷贝）。

遗留（不影响本轮接入）：两个 kernel 都不读 `conv_state.stride`，非连续视图靠 torch 侧自动拷贝兜底；
`idx=-1` 在 prefill kernel 里被断言拒绝（update kernel 是跳过 `-1`），两者空槽约定仍不一致。

## 优先级
P0

## 目标版本与期望时间
GLM-5.3-Flash 适配 Phase 2（性能阶段）。建议与 #02、#03、#04 一起在长上下文（≥32K prefill）性能验收前给出，具体里程碑待算子组排期确认。
## 算子名称
修复 `torch.dlc_kda_causal_conv_prefill`（采用新算子；修复 N 分块卡死 + 放行 -1 + 去掉 host D2H 同步）

## 背景与功能说明
使用场景：GLM-5.3-Flash 有 34 层 KDA 线性注意力，每层对 q/k/v 投影各做一次 kernel=4 的深度可分离短卷积（DWConv），prefill 阶段按变长 batch 计算，并把每条序列末尾的 `kernel-1` 个历史 token 写回 conv state 页。
当前问题：插件全部用纯 PyTorch 组合实现（`vllm_cl/patch/patch_kda.py` 的 `causal_conv1d_fn_dlc` → `_native_conv_sequence`）。实现方式是在 Python 里对 token 逐位循环，每一步 `torch.cat` 拼滑窗再做逐元素乘加，等价于：

```python
for t in range(length):
    window = torch.cat((hist, x[:, t:t+1]), dim=-1)   # [dim, K]
    y[:, t] = silu((window * weight).sum(-1) + bias)
    hist = torch.cat((hist[:, 1:], x[:, t:t+1]), dim=-1)
```

TP8 下 dim=3072、prefill 4096 token、34 层时会产生约 14 万次小 kernel 与同量级的 `cat`，实测是当前 prefill 的主要瓶颈，且无法进入 CUDA graph（Python 循环 + 动态 shape）。
目标功能：单个（或少量）kernel 完成定长滑窗深度卷积 + 变长序列切分 + conv state 原位写回，并支持 cache 页映射与 pad slot 跳过。
非目标：不做 q/k/v 投影 GEMM 的融合（仍由 Linear 承担）；不做 MTP/spec 的多 token 更新路径（见 `dlc_kda_causal_conv_update`）。

## 期望 API 与函数签名
> **2026-09-22 晚更新：本条改为「提交并采用新算子」**（详见文件顶部与 `给算子组_01_02_修复要点.md`）。
> 下面这段签名是算子组实际交付的形式（初稿拟定的 5 参数版本已被取代），保留作接口对照；
> （见「需求分类」段与「算子名称」），**不需要算子组提供这个新 API**。

```python
def causal_conv1d_fn_prefill(
    x: torch.Tensor,                              # [dim, total_tokens]  bf16
    weight: torch.Tensor,                         # [dim, 1, kernel]    bf16, depthwise
    bias: torch.Tensor | None,                    # [dim]               bf16
    conv_states: torch.Tensor,                    # [num_pages, dim, kernel-1] bf16, 原位写回
    query_start_loc: torch.Tensor,                # [num_seqs + 1]      int32
    cache_indices: torch.Tensor | None = None,    # [num_seqs]          int32
    has_initial_state: torch.Tensor | None = None,# [num_seqs]          bool
    activation: str = "silu",
    pad_slot_id: int = -1,
) -> torch.Tensor:                                # [dim, total_tokens] bf16
```

## 参数与返回值
输入：
- `x`：`[dim, total_tokens]`，BF16，contiguous，dim 通道、total_tokens 拼接后的变长 token。
- `weight`：`[dim, 1, kernel]`，BF16，深度卷积权重（每通道独立，无跨通道混合）。
- `bias`：`[dim]`，BF16，可为 None。
- `conv_states`：`[num_pages, dim, kernel-1]`，BF16，按物理页索引的卷积历史，**原位更新**。
- `query_start_loc`：`[num_seqs+1]`，int32，前缀和形式的序列边界，单调不减，`query_start_loc[0]==0`。
- `cache_indices`：`[num_seqs]`，int32，第 s 条序列使用的物理页号；None 时退化为页号 = s。
- `has_initial_state`：`[num_seqs]`，bool，False 表示该序列从零状态开始（忽略页内旧历史）；None 视为全 True。

属性：
- `activation`：str，默认 `"silu"`；支持 `"silu"` 与 `"none"`。
- `pad_slot_id`：int，默认 `-1`；`cache_indices[s] == pad_slot_id` 时该序列只计算输出、不读写 state。

返回值：
- `y`：`[dim, total_tokens]`，BF16，与 `x` 同形同 dtype；不修改 `x`。

边界：空序列（`end <= start`）跳过；`total_tokens == 0` 返回空张量。

## 功能语义
计算公式（K = kernel，hist 为页内历史，长度 K-1）：

```text
win[:, j] = x[t-K+1+j]      j = 0..K-1        # 不足处补 hist / 0
y[:, t]   = activation( sum_j(win[:, j] * weight[:, 0, j]) + bias )
hist      = win[:, 1:]                        # 推进一位
```

逐序列伪代码：

```text
for s in range(num_seqs):
    start, end = query_start_loc[s], query_start_loc[s+1]
    if end <= start: continue
    page = cache_indices[s] if cache_indices is not None else s
    if page == pad_slot_id: continue          # 只算输出
    hist = conv_states[page]                  # [dim, K-1]
    if has_initial_state is not None and not has_initial_state[s]: hist = 0
    for t in start..end-1:
        win = concat(hist, x[:, t])           # [dim, K]
        y[:, t] = activation(sum(win * weight[:, 0, :], -1) + bias)
        hist = concat(hist[:, 1:], x[:, t])
    conv_states[page] = hist
```

broadcasting / padding 规则：序列之间不共享 state，页内历史按物理页独立；无 mask。
in-place / alias 规则：只写 `conv_states`，不修改 `x`；要求 `x` 与 `conv_states` 不重叠。
空 / 零长度：`total_tokens == 0` 或 `num_seqs == 0` 直接返回空输出。
异常输入：`query_start_loc` 非单调、越界，`cache_indices` 越界，`x.shape[0] != weight.shape[0]` 均返回错误。
确定性：无随机，逐元素结果确定。
累加精度：窗口内乘加与 bias 用 FP32 累加，最后一次性 cast 回 BF16。

## 支持范围与约束
支持的 dtype：`x`/`weight`/`bias`/`conv_states` 均为 BF16；内部 FP32 累加。（若算子组支持 FP32 输入请一并声明。）
支持的 shape 及动态范围：
- `dim` 动态，GLM-5.3-Flash 实际取值 `3 * H_local * 128`：TP8 → 3072，TP4 → 6144，TP2 → 12288；不要写死。
- `kernel` 常见 4，建议支持 2/3/4 泛化；`kernel-1` 即 state 宽度。
- `total_tokens` 动态，建议覆盖到 32K~64K；`num_seqs` 动态。
数值范围：输入为 BF16 正常范围；`activation="silu"` 需稳定处理大正值（silu 不溢出 BF16 输出）。
layout / stride：`x` 要求 contiguous 的行主序 `[dim, total]`；`conv_states` 要求最后一维连续。
device：DLC 单卡；不涉及跨卡通信。
执行模式：需支持 CUDA graph 捕获（shape 静态时）与 eager。
其他限制：不支持 `x` 与 `conv_states` alias；不支持 grouped >1 的跨通道卷积。

## 精度要求
Golden / reference 实现：`vllm_cl/patch/patch_kda.py` 的 `_native_conv_sequence`（PyTorch，FP32 累加），可在 CPU 或 DLC 上运行。
输入数据分布：`x ~ N(0, 1)`、`weight ~ N(0, 1/K)`，另加极端 case（全 0、超阈值 ±1e4）。
样本数量：每类 shape ≥ 200 组随机样本，固定随机种子（如 0/1/2）。
比较 dtype：BF16 输出对 FP32 reference 计算误差，同时比较写回后的 `conv_states`。
atol / rtol：输出 `atol=2e-2, rtol=2e-2`；`conv_states` 按同标准逐元素比较。
必须逐元素一致的场景：`activation="none"` 且输入为可精确表示的 BF16 值时，要求与 reference bit-exact。
NaN / Inf / 极值 / 零值：全 0 输入输出必须为 `bias` 经 activation 的结果；±Inf 输入不做特殊要求但不得崩溃。
累加与中间计算精度：窗口乘加与 bias 用 FP32；state 存储按 BF16 写回。

## 测试范围
- [ ] 基本功能与 reference 对齐
- [ ] 多 dtype 精度
- [ ] 静态 shape 与动态 shape
- [ ] 最小值、最大值、空 tensor 等边界
- [ ] 非 contiguous、特殊 stride 或 layout
- [ ] 非法参数和异常处理
- [ ] 单卡
- [ ] 多卡/分布式
- [ ] 长时间稳定性
- [ ] 性能基准与性能回归

具体测试用例、shape、数据范围和预期结果：
1. 基本功能：单个序列 prefill（T=1/16/1024/4096）、多序列变长混合（如 [1, 7, 512, 2048]）、`has_initial_state=False` 冷启动、`cache_indices` 乱序映射。
2. 动态 shape：`total_tokens ∈ {1, 2, 3, 5, 255, 256, 257, 4096}`；`num_seqs ∈ {1, 8, 64}`；`dim ∈ {3072, 6144}`。
3. 边界：空序列、`pad_slot_id` 命中、`bias=None`、`kernel=2/3/4`。
4. 精度：`atol/rtol` 达标；`conv_states` 写回与 reference 一致；连续两次调用（第二次带初始 state）结果一致。
5. layout：非 contiguous 输入给出明确报错或内部处理。
6. 异常：`query_start_loc` 非单调、`cache_indices` 越界、`x.shape[0] != weight.shape[0]` 均需报错。
7. 性能：与当前 native 实现对比，prefill 4096 token × 34 层端到端加速比（目标 ≥ 10×），并给出单 kernel 时延。

## 验证环境与依赖版本
DLC 硬件：8 卡，设备名 `/dev/cltech0-7`（TP=8，无 PP）
vLLM commit：`f5361fb09f`
vLLM DLC（vllm-cl）commit：`a14fc5b`
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：`VLLM_USE_DLC_COL_MAJOR_MATMUL=1`
模型：GLM-5.3-Flash-Mix（AWQ 4-bit，`linear_num_heads=64`、`linear_head_dim=128`、`short_conv_kernel_size=4`、`gate_lower_bound=-5.0`）

## 验收标准
- [ ] 插件侧按实测约束构造输入：`x=[dim,T]` 且通道维连续、`weight` 为 2D `[dim,K]`（KDA 的 `[dim,1,K]` 需 squeeze）、`conv_states` 通道维 `stride==1`、bf16 `storage_offset` 为偶数、APC 相关参数为 None。
- [ ] 接入后 prefill 输出与 native reference 一致（实测 `out_err=1.953e-03`、`state_err=0.00e+00`）。
- [ ] 不修改现有算子：本条不产生算子组接口/行为变更。
- [ ] 性能：34 层短卷积相对「逐 token Python 循环」的耗时下降，记录 profile 数据。
## Reference、日志与附件
Reference 实现：`vllm_cl/patch/patch_kda.py:21`（`_native_conv_sequence`）、`:31`（`causal_conv1d_fn_dlc`）。
调用点：`vllm/models/glm5next/nvidia/kda.py:462`（`causal_conv1d_fn(...).transpose(0,1)`）。
上游语义参考：`vllm/model_executor/layers/mamba/ops/causal_conv1d.py` 的 `causal_conv1d_fn`（Triton）。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md` §5.4 算子 `dlc_kda_causal_conv_prefill`。
日志 / profile：`vllm-cl/glm53_flash.log`（含 prefill 阶段每层调用次数与耗时）。
