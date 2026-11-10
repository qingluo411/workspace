<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22 定稿）：需扩展已有算子能力（不新增独立算子） -->
<!-- 需求分类（2026-09-22 二次，算子组交付新算子后实测）：改为「需修复算子功能或精度」
     dlc_kda_recurrent_decode 的 state 数值正确（按 fla 语义逐项对齐 state_err≈3e-5，T=1/2/4 一致），但 out 逐位恒为 0
     （kernel dlc_kda_recurrent_decode.cpp:189 把常量 0.0f 写进 torch 回读的那半；二次调用 state 仍在变化）；
     state 布局原为 [S,H,Dk,Dv]（行=Dk），与 fla / 插件 native 的 [pages,H,V,K] 互为转置；
     **2026-09-22 算子组已同意统一为 [S,H,Dv,Dk]（与 fla/native 一致）**，新布局的 fp32 参考实现见
     refs/kda_reference_dv_dk.py；另 num_accepted_tokens 形参被接受但未生效（总是提交整窗）
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：部分覆盖（建议改为「扩展已有算子能力」）
     证据：fused_recurrent_gated_delta_rule 在 per-head gate 下实测 err 7.77e-04，已支持 ssm_state_indices / num_accepted_tokens / qk-l2norm；但逐通道 gate 被拒；dlc_vllm_recurrent_gated_delta_rule_packed_decode 内部用标准 GDN softplus gate，与 GLM5 bounded sigmoid gate 语义不同
     建议：改为扩展 fused_recurrent_gated_delta_rule：支持逐通道 gate + bounded safe gate，复用其 state/spec 能力 -->

<!-- ======================================================================
     2026-09-28 状态：已修复、已接入、真机验证通过（本条 issue 可关闭）
     ----------------------------------------------------------------------
     根因（两处，均已修）：
       1) kernel 侧：out 恒 0 / num_accepted_tokens 不生效 / state 布局转置
          —— 算子组在 feat/dlc_kda_recurrent_decode 修复，state 统一为 [S,H,Dv,Dk]。
       2) aten 侧：`dlc_kda_recurrent_decode` 内部声明单个 `DLCScalar *scalars_`（按
          scalars_[0..2] 读），而 `next_arg<DLCScalar>(n)` 每次调用都新开一条 arg record；
          原代码三次独立 `.scalar()` → kernel 只读到 scalars_[0]，scale 被读成 0 → q 清零
          → out≡0、has_accepted 恒假。修复：合并成一次
          `kd.scalar(lower_bound, kernel_scale, has_num_accepted)`。
          （pytorch 侧 `aten/src/ATen/native/dlc/KdaRecurrentDecode.cpp`，未提交，已在本机重编验证。）
     接入：`vllm_cl/patch/patch_kda.py:fused_recurrent_kda_dlc` decode 分支走新算子
          （`_recurrent_decode_new_op`），开关 `DLC_KDA_RECURRENT_DECODE_OP`（默认 dlc）。
     真机：8 worker × 34 层 decode 零 fallback，0.49–0.60 ms/call（34 层 ~17–20 ms/步）；
          单请求与 3 并发批解码输出正确；回归 `tests/test_glm5_kda_recurrent_decode_op.py`（17 例）。
     参考布局：CPU 参考已改为 [H,Dv,Dk]（`pytorch/test_tpu/test_aten/test_attention/test_kda_recurrent_decode.py`；
          插件侧同构参考 `refs/kda_reference_dv_dk.py`，其中 accepted==0 修正为回滚到进入本步前的 state）。
     ====================================================================== -->

# [Operator Request] dlc_kda_recurrent_decode（KDA 单步/多步递推 decode）

## 需求类型
修复算子功能或精度

**需求分类：需扩展已有算子能力（不新增独立算子）**
- 现有算子：`torch.fused_recurrent_gated_delta_rule(q, k, v, g, beta=None, scale=None, initial_state=None, inplace_final_state=True, cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None, use_qk_l2norm_in_kernel=False)`；另有 `torch.dlc_vllm_recurrent_gated_delta_rule_packed_decode(mixed_qkv, a, b, A_log, dt_bias, scale, ssm_state_indices, use_qk_l2norm=False)`。
- 实测（dlc:0）：per-head gate 下与 fp32 递推一致（`out_err=7.77e-04`、`state_err=1.88e-03`），并已支持 `ssm_state_indices`、`num_accepted_tokens`、in-kernel qk-l2norm —— KDA decode 需要的 state 页索引与 speculative 回滚能力已经具备。
- 缺口：
  1. **逐通道 gate `[1, T, H, K]` 被拒绝**（`g must have shape [T, HV]`）；KDA 的衰减是 `h[k, v] *= exp(g[t, h, k])` 的逐 K 通道衰减。
  2. `dlc_vllm_recurrent_gated_delta_rule_packed_decode` 在 kernel 内按标准 GDN 生成 gate（`-exp(A_log) * softplus(a + dt_bias)`），与 GLM5 的 bounded safe gate `lower_bound * sigmoid(exp(A_log) * (raw_g + g_bias))` 语义不同，**不能直接复用**。
  3. GLM5 传的是 raw bf16 `beta`，需要 kernel 内 `sigmoid(beta)`。
- 落地方式：扩展 `fused_recurrent_gated_delta_rule` 支持「逐通道 gate + bounded safe gate + 可选 sigmoid_beta」，即可覆盖 KDA decode 与 MTP verify（含 `num_accepted_tokens`），无需新算子。
- 复现脚本：`probes/probe_kda_recurrent.py`。
- 本次调整（2026-09-22）：需求类型「新增算子」→「扩展已有算子能力」；优先级 P0 不变；算子名称改为扩展 `torch.fused_recurrent_gated_delta_rule`（原提名的 `dlc_kda_recurrent_decode` 合并入该算子）。
- 本次调整（2026-09-22 二次，算子组交付新算子后实测）：算子组已交付专用算子 `dlc_kda_recurrent_decode`。
  2026-09-22 晚复核（修正初版两处不准确表述）：**state 递推完全正确** —— 与 fla 语义（`h=[H,Dk,Dv]`、衰减 `exp(g)` 沿 Dk）
  逐项对齐后 `state_err≈3e-5`（bf16 级），T=1/2/4 一致；但 **`out` 逐位恒为 0**（`(out==0).all()==True`、`out.abs().max()==0.0`），
  已定位到 kernel 写回行把常量 `0.0f` 打进 torch 回读的那半；state 布局 `[S,H,Dk,Dv]`（行=Dk）与插件 native 的
  `[pages,H,V,K]`（行=Dv）互为转置（实测语义一致）；**2026-09-22 算子组已同意统一为 `[S,H,Dv,Dk]`，插件侧零转换**，新参考实现见 `refs/kda_reference_dv_dk.py`；`num_accepted_tokens` 形参被接受、
  但 **state 仍按整窗（T 个 token）提交，未按 accepted 截断**。故需求类型改为「**修复算子功能或精度**」，
  算子名称改回 `dlc_kda_recurrent_decode`；优先级 P0 不变。


## 2026-09-22 新算子实测（算子组已交付 `dlc_kda_recurrent_decode`）

算子组已提交专用算子 `dlc_kda_recurrent_decode`
（`aten/src/ATen/native/dlc/KdaRecurrentDecode.cpp`，kernel `custom_dlc_kda_recurrent_decode`），
故本条的诉求从「扩展 `fused_recurrent_gated_delta_rule`」改为「**修复这个已交付但输出不可用的新算子**」。
实测脚本：`probes/probe_new_kda_recurrent_decode.py`（基础用例）与 `probes/probe_new_kda_recurrent_decode_fla_ref.py`
（2026-09-22 晚复核：按 fla 语义逐项对齐 + 受控判决用例 + `out` 逐位检查；可直接粘贴的完整 Issue
正文见 [`给算子组_03_04_修复要点.md`](给算子组_03_04_修复要点.md)。初版 `..._layout.py` 的参考实现
把衰减轴写错、结论不可用，已改名重写）。

**实测结论：state 递推完全正确，但 `out` 恒为全 0 —— 输出路径坏了。**

| 用例（H=8, Dk=Dv=128） | state | out |
|---|---|---|
| plain decode B=4（含 null slot 0） | 正确（null slot 输出 0、state 不变） | 参考量级 0.0157，实测 **0** |
| spec T=2/T=4（`accepted=[1,2]`/`[0,3]`）、变长 [3,1,2] | 与 fp32 递推一致（相对误差 ~1%） | **0** |
| 判决性用例：`q=k=e3, v=10, α=1, β=1, h0=0` | `h[3,:]=9.9994`（期望 10 ✓，其余行 0 ✓） | 期望 `10×scale=0.884`，实测 **0** |
| `out` 预填 7.0 再调用 | — | 被 kernel **覆写成 0.0**（说明写操作发生了，写进去的值是 0） |

**根因（已定位到代码，可直接改）**：kernel 的输出写回把常量 `0.0f` 当成了要写的值 ——

```c
// dlc_kda_recurrent_decode.cpp:189（错误的写回）
v_f32_st_tnsr_st_msk(0, raw_vmem, 1, 1, __$F(float_to_bfloat16(v_u32_move_f(0.0f), output)));
//                                                                       ^^^^^ 这一半是 torch 回读的那半 → out 全 0
// 相邻的 state 写回是正确的：state_row += broadcast_lane(k, key) * delta; v_f32_st_tnsr_st_msk(0, row, 1, 1, state_row);
```

因此实测现象是「**state 每次都正确更新（二次调用仍在变），`out` 恒为 bf16 的 0.0**」，而不是打包顺序随机错位：
把 `output` 放进 torch 回读的那一半（两个实参互换，或两半都放 `output`）即可修复；修复后请用
`probes/probe_new_kda_recurrent_decode_fla_ref.py` 复跑（该脚本断言 `out != 0` 且与 fp32 参考一致）；
**布局改为 `[S,H,Dv,Dk]` 后请改用 `refs/kda_reference_dv_dk.py`** —— 旧脚本的参考实现是交付版 `[Dk,Dv]` 布局，届时会不匹配。

**state 布局（2026-09-22 定稿：统一为 `[S, H, Dv, Dk]`，行 = value 维）**：
- 交付版 kernel 用的是 `[S_slot, H, Dk, Dv]`（行 = key 维、衰减沿 Dk），与 fla / 插件 native 的
  `[pages, H, V, K]` 互为转置；因为 Dk=Dv=128，两者形状都合法，**接错不报错、只会静默把 key/value 轴用反**。
  判决用例（不依赖参考实现）：`state=0, k=e5, v=e3, β=1` → 交付版只有 `[dk=5, dv=3]` 一个非零元；
  `state=1, 只剩衰减` → 沿 dim2 变化、dim3 不变，逐值等于 `exp(-5·sigmoid(g_bias))`。
- **算子组 2026-09-22 同意把 `recurrent_state`（连同 #03 的 `initial_state`/`final_state`）统一为
  `[S,H,Dv,Dk]`**：行 = value 维、衰减 `exp(gate)` 沿 Dk **列**乘。这样插件侧可零转换地把 KDA state 页
  直接传进来（与 `_run_kda`、上游 fla 一致）。
- kernel 侧若更习惯「Dv 在向量 lane」，可在载入/写回时于 VMEM 内做一次 128×128 转置：搬运量不变
  （`state_words = 128*128`），**不要额外走一遍 HBM**。
- 新布局 fp32 参考实现：`refs/kda_reference_dv_dk.py`（`recurrent_decode_ref` / `chunk_prefill_ref`，两者互相自检等价；
  用法见 `refs/README.md`）。

**其他接口差异（修复时请一并处理/确认）**：
- `A_log` 要 1D `[H]` fp32，而模型里是 `[1,1,H,1]`；`g_bias` 要 `[H,Dv]` fp32，模型里 `dt_bias` 是扁平 `[H*Dv]`；
  注意同批的 `dlc_kda_chunk_prefill` 要的是 `[D,H]`（相反），两个算子约定不一致；
- `lower_bound` 必须精确等于 `-5.0`（硬校验）；
- 所有输入必须 contiguous 且 `storage_offset==0`，物理行必须是 512 B（`dlc_dim0_padded_*dlc_layout_==512`）；
- `state_indices` 支持 `[B]` 与 `[B,next_n]`。
- **`num_accepted_tokens` 未生效（spec 回滚语义缺失）**：传入 `acc=[2]`（T=4）时，state 与「整窗 T=4 提交」吻合（`state_err=3.0e-05`），
  与「第 2 个 token 后快照」差 `2.6e-01` → kernel 目前总是提交整窗。插件现状不支持 spec（`fused_recurrent_kda_dlc` 直接
  `raise NotImplementedError`），故不影响当前接入；但既然签名已声明支持，修复时请一并实现（或从签名中删掉以免误用）。

**处置**：本条需求类型由「扩展已有算子能力」改为「**修复算子功能或精度**」，算子名称改回 `dlc_kda_recurrent_decode`。
修复内容只有一处：kernel `dlc_kda_recurrent_decode.cpp:189` 的输出写回（把 `output` 放进 torch 回读的那半）；
同时请（a）把 `recurrent_state` 改为 `[S,H,Dv,Dk]`（2026-09-22 已确认，见上）、（b）实现或删除 `num_accepted_tokens`。修复前插件继续用逐 token 递推
（`fused_recurrent_kda_dlc`，不支持 spec 的 `num_accepted_tokens`，见 #02 的记录）。

## 优先级
P0

## 目标版本与期望时间
GLM-5.3-Flash 适配 Phase 2（性能阶段）。建议在长上下文（≥32K prefill）性能验收前提供，具体里程碑待算子组排期确认。

## 算子名称
修复 `torch.dlc_kda_recurrent_decode`（缺陷单点：`dlc_kda_recurrent_decode.cpp:189` 输出写回把常量 0.0f 写进 torch 回读的那半；另需实现 num_accepted_tokens）

（2026-09-22 之前的目标是「扩展 `torch.fused_recurrent_gated_delta_rule`」，现已由专用算子替代，仅当修复方案放弃时才回到扩展路线。）

## 背景与功能说明
使用场景：KDA 34 层在 decode（含 MTP verify）阶段的递归状态更新与 read，是每 token 必经路径。
当前问题：native 实现 `fused_recurrent_kda_dlc`（`vllm_cl/patch/patch_kda.py:134`）是 Python 逐 token 循环 + `bmm`，并且**显式不支持 speculative**（`num_accepted_tokens is not None` 直接 `NotImplementedError`，`patch_kda.py:135`）。decode 每步 34 层 × 每层若干小算子，launch 开销和 Python 循环是主要瓶颈，且 MTP 无法启用。
目标功能：单 kernel 完成每请求 1~8 个 token 的递推，state 原位更新，支持 gate/beta 的 in-kernel 计算与按接受数提交。
非目标：prefill 的 chunked 路径（见 03 号）。

## 期望 API 与函数签名
> **本条是「扩展已有算子」**：算子名保持 `torch.fused_recurrent_gated_delta_rule`，只是并入了逐通道 gate、
> bounded safe gate 与可选 `sigmoid(beta)`；下面的签名是「新增项并入后的形态」，供算子组对照实现。

```python
def fused_recurrent_gated_delta_rule(  # 现有算子，仅新增参数/行为
    q: torch.Tensor,                  # [1, T, H, Dk] bf16
    k: torch.Tensor,                  # [1, T, H, Dk] bf16
    v: torch.Tensor,                  # [1, T, H, Dv] bf16
    g: torch.Tensor,                  # [1, T, H, Dv] bf16, raw gate logit
    beta: torch.Tensor,               # [1, T, H]     bf16, raw（kernel 内 sigmoid）
    initial_state: torch.Tensor,      # [num_pages, H, Dv, Dk] fp32, 原位更新
    cu_seqlens: torch.Tensor,         # [num_seqs + 1] int32
    ssm_state_indices: torch.Tensor | None = None,  # [num_seqs] int32
    num_accepted_tokens: torch.Tensor | None = None,# [num_seqs] int32
    out: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    sigmoid_beta: bool = True,
    a_log: torch.Tensor | None = None,   # [H] fp32
    g_bias: torch.Tensor | None = None,  # [H, Dv] fp32
    compute_gate: bool = True,
    lower_bound: float = -5.0,
) -> tuple[torch.Tensor, torch.Tensor]
```

## 参数与返回值
输入 / 属性：
- `q/k/v/g/beta/cu_seqlens`：语义同 03 号需求；decode 的 `beta` 是 **raw logit**，由 `sigmoid_beta=True` 在 kernel 内 sigmoid。
- `initial_state`：`[num_pages, H, Dv, Dk]` FP32，**按物理页索引，原位更新**。
- `ssm_state_indices`：`[num_seqs]` int32，每序列的物理页；负值（PAD_SLOT）表示只算输出不写状态。
- `num_accepted_tokens`：`[num_seqs]` int32，spec 路径下每请求接受数；未接受 token 的输出仍需计算，但不推进/提交状态。
- `out`：可选输出缓冲，形状 `[1, T, H, Dv]`；提供时原位写。
- `a_log/g_bias/compute_gate/lower_bound`：`gate = lower_bound * sigmoid(exp(A_log)*(g + g_bias))`。
- `sigmoid_beta`：默认 True。

返回值：
- `(out, initial_state)`；`out` 为 `[1,T,H,Dv]` BF16，状态已原位更新。

边界：`ssm_state_indices` 负值跳过；`num_accepted_tokens=0` 时状态保持不变；重复页号需报错（native 已有该校验）。

## 功能语义
逐 token 递推（与 03 号需求同公式），差异在于：

```text
beta_t = sigmoid(beta_raw_t)        # sigmoid_beta=True
gate_t = lower_bound * sigmoid(exp(A_h) * (g_t + g_bias_h))
alpha  = exp(gate_t)
S      = diag(alpha) @ S
delta  = beta_t * (v_t - S^T k_t)
S      = S + k_t delta^T
o_t    = (S^T q_t) * scale
```

spec 路径：

```text
for t in range(T):
    if t < num_accepted_tokens[s]:
        commit S       # 只有被接受的 token 推进并写回状态
    emit o_t           # 所有 token 都要输出，供 rejected 位置后续使用
```

in-place：`initial_state` 原位写，`out` 若提供则原位写。
空 / 零长度：`T == 0` 返回空输出与不变状态。
异常：页号越界、`ssm_state_indices` 长度与 `num_seqs` 不符、重复页号报错。
确定性：无随机；累加 FP32。

## 支持范围与约束
支持的 dtype：`q/k/v/g/beta` BF16；`initial_state` FP32；`a_log/g_bias` FP32。
支持的 shape：`T` 支持 1（plain decode）与 2/4/6/8（MTP draft）；`H` 为 TP 分片头数；`Dk=Dv=128`。
数值范围：`gate ∈ [-5, 0)` → `alpha ∈ (0.0067, 1]`，无溢出风险。
layout：packed `[1,T,H,*]`；`initial_state` 最后一维连续。
device：DLC 单卡。
执行模式：**必须支持 CUDA graph**（decode 主路径）。
其他限制：与 03 号需求共用数学定义，两条路径在 `T=1` 时必须数值一致（这是关键的交叉验收项）。

## 精度要求
Golden / reference：`vllm_cl/patch/patch_kda.py:93`（`_run_kda`，FP32）。
输入分布：同 03 号；额外覆盖 `num_accepted_tokens ∈ {0, 1, ..., T}`。
样本数量：≥ 200 组 / 每类 shape。
比较 dtype：BF16 输出 + FP32 状态。
atol / rtol：输出 `2e-2`；状态 `1e-3`。
必须逐元素一致：`num_accepted_tokens=0` 时状态与调用前 bit-exact；`T=1` 时与 chunk kernel 输出 bit-exact。
极值：`alpha` 取下界 0.0067、上界 1.0 时稳定。
累加精度：全程 FP32。

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
1. 基本功能：T=1，B ∈ {1, 8, 64, 256}；带 / 不带 `out`；带 / 不带 `ssm_state_indices`（None 时序号即页号）。
2. spec：T ∈ {2,4,6,8}，接受数覆盖 0 / 中间 / 全部。
3. 交叉一致性：与 03 号 chunk kernel 在 T=1 输出逐元素一致。
4. 边界：PAD_SLOT 负索引、`T=0`。
5. 精度 / 异常 / CUDA graph 捕获。
6. 性能：B=64、34 层 decode 单步总耗时，目标 < 2ms，并给出相对 native 加速比。

## 验证环境与依赖版本
DLC 硬件：8 卡，设备名 `/dev/cltech0-7`（TP=8，无 PP）
vLLM commit：`f5361fb09f`
vLLM DLC（vllm-cl）commit：`a14fc5b`
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：`VLLM_USE_DLC_COL_MAJOR_MATMUL=1`
模型：GLM-5.3-Flash-Mix（AWQ 4-bit，`linear_num_heads=64`、`linear_head_dim=128`、`short_conv_kernel_size=4`、`gate_lower_bound=-5.0`）

## 验收标准
- [ ] API、参数校验和错误行为符合约定
- [ ] 规定范围内的功能和精度测试通过
- [ ] 需求中明确列出的特殊运行场景通过
- [ ] 文档、测试和示例已提交
- [ ] 接入 vllm-cl 后，GLM-5.3-Flash 在 DLC 上 prefill/decode 数值与 PyTorch reference 一致

## Reference、日志与附件
Reference 实现：`vllm_cl/patch/patch_kda.py:93` / `:134`。
调用点：`vllm/models/glm5next/nvidia/kda.py:510`（spec）、`:577`（plain decode）。
上游语义参考：`vllm/third_party/flash_linear_attention/ops/kda.py` 的 `fused_recurrent_kda`。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md` §5.2/§5.4。
