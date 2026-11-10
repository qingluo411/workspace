<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22 定稿）：需扩展已有算子能力（不新增独立算子） -->
<!-- 需求分类（2026-09-22 二次，算子组交付新算子后实测）：改为「需修复算子功能或精度」
     算子组已交付 dlc_kda_chunk_prefill，但（a）host wrapper 5 维张量用 4 阶 permute → 每次调用必抛异常；
     （b）launch 出去的 kernel 会把设备卡死（dlc:2/dlc:3 均需复位）；（c）kernel 就地改写输入 beta_logits。
     修复前插件继续用逐 token 递推；修复后需重新做数值对齐。-->
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：部分覆盖（建议改为「扩展已有算子能力」）
     证据：chunk_gated_delta_rule 的 delta-rule 核心实测正确（T=2 与 fp32 递推 err 1.02e-04），但只接受 per-head 标量 gate [T,HV]；GLM5 的逐通道 gate [1,T,H,D] 被参数校验拒绝（got dim 4 / batch size must be 1）
     建议：改为扩展 chunk_gated_delta_rule：支持逐通道 gate + 融合 bounded safe gate（lower_bound*sigmoid(exp(A_log)*(raw_g+g_bias))）与 chunk 内累计衰减，而非新增独立 chunk 算子 -->

# [Operator Request] dlc_kda_chunk_prefill（KDA chunked prefill）

## 需求类型
修复算子功能或精度

**需求分类：需扩展已有算子能力（不新增独立算子）**
- 现有算子：`torch.chunk_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=True, cu_seqlens=None, head_first=False, use_qk_l2norm_in_kernel=False)`（实现：`aten/src/ATen/native/dlc/Chunk_gated_delta_rule.cpp`）。
- 实测（dlc:0）：delta-rule 核心数值正确——per-head gate 下与 fp32 逐 token 递推一致（`out_err=1.02e-04`、`state_err=8.77e-04`），且 `g` 的语义是「逐 token 的 log 衰减」（`decay = exp(g)`）。
- 缺口（需要算子侧补齐，插件无法绕开）：
  1. **只接受 per-head 标量 gate `[T, HV]`**；KDA/GLM5 的 gate 是逐通道 `[1, T, H, D]`（D = head_dim = 128，衰减乘在 state 的 K 维上）。实测报 `g must be either flattened or batched, but got dim 4` / `g batch size must be 1 when cu_seqlens is used`。
  2. 需要融合 **bounded safe gate**：`gate = lower_bound * sigmoid(exp(A_log) * (raw_g + g_bias))`（GLM5 `kda_gate_cumsum_fwd_kernel` 语义）；现有 `fused_gdn_gating` 是标准 GDN 的 `-exp(A_log) * softplus(a + dt_bias)`，两者不等价。
  3. chunk 路径还需要 **chunk 内累计衰减**（gate 先做块内 `cumsum` 再乘 `RCP_LN2`，供 `exp2` 内核使用）。
- 落地方式：请把上述三点并入 `chunk_gated_delta_rule`（例如接受逐通道 `g`，并提供 `use_kda_gate`/`lower_bound` 等参数）；落地后插件即可删除 `vllm_cl/patch/patch_kda.py` 里的逐 token Python 循环，本需求不再需要新算子。
- 复现脚本：`probes/probe_kda_chunk_reference_compare.py`、`probes/probe_kda_chunk_gate_convention.py`、`probes/probe_kda_chunk_perchannel_rejected.py`。
- 本次调整（2026-09-22）：需求类型「新增算子」→「扩展已有算子能力」；优先级 P0 不变；算子名称改为扩展 `torch.chunk_gated_delta_rule`（原提名的 `dlc_kda_chunk_prefill` 合并入该算子）。
- 本次调整（2026-09-22 二次，算子组交付新算子后实测）：算子组已交付专用算子 `dlc_kda_chunk_prefill`，
   但 host wrapper 每次调用必抛 `permute` 异常、且 launch 出的 kernel 会卡死设备（详见下节），
   故需求类型改为「**修复算子功能或精度**」，算子名称改回 `dlc_kda_chunk_prefill`；优先级 P0 不变。


## 2026-09-22 新算子实测（算子组已交付 `dlc_kda_chunk_prefill`）

算子组已在 pytorch `test-glm53flash` 分支提交专用算子 `dlc_kda_chunk_prefill`
（host wrapper `aten/src/ATen/native/dlc/FusedOps.cpp:780-845`，kernel `custom_dlc_kda_chunk_prefill`），
故本条的诉求从「扩展 `chunk_gated_delta_rule`」改为「**修复这个已交付但完全不可用的新算子**」。

**实测结论：该算子当前 100% 调用失败，且会把设备卡死（脚本 `probes/probe_new_kda_chunk_prefill.py`）。**

### state 布局定稿（2026-09-22）：`[B, H, Dv, Dk]`

算子组已同意把 `initial_state` / `final_state` 统一为 **`[B,H,Dv,Dk]`**（行 = value 维、衰减沿 Dk 列乘），
与 fla / 插件 native 的 `[pages,H,V,K]` 一致，插件侧零转换。kernel 内部若更习惯「Dv 在向量 lane」，
可在 VMEM 内做一次 128×128 转置（搬运量不变，别额外走 HBM）。
新布局的 fp32 参考实现：[`refs/kda_reference_dv_dk.py`](refs/kda_reference_dv_dk.py) 的 `chunk_prefill_ref()`
（自带与逐 token 递推的等价自检）。修复时请同步改掉 C++ 里
`// Keep the public PyTorch API in [1, N, H, D] / [B, H, Dk, Dv] layout` 的注释与 `permute` 顺序；
公开签名里 `g_bias` 仍是 `[Dk,H]`（内部 `permute({1,0})` 成 `[H,Dk]`）。

> 2026-09-22 晚：脚本已改成两段式 —— `--static`（默认，**不占设备**，用纯 torch 复现同一条 `permute` 异常）、
> `dlc:N --wedge-device`（真机复现，**会卡死该卡**）。可直接粘贴的完整 Issue 正文（含脚本全文、根因代码行、
> 验收标准）见 [`给算子组_03_04_修复要点.md`](给算子组_03_04_修复要点.md)；卡死判定用 `probe_device_health.py`。

1. **host wrapper 每次调用都抛异常**（与输入数据无关，任何 shape 都必现）：
   ```
   RuntimeError: TensorShape.cpp:1417(_permute_size_stride_estimation)
     permute(sparse_coo): number of dimensions in the tensor input does not match
     the length of the desired ordering of dimensions
     i.e. input.dim() = 5 is not equal to len(dims) = 4
   ```
   原因：`FusedOps.cpp` 里 `auto out_dlc = at::empty_like(v_dlc);`（`v_dlc` 已是 5 维，
   因为前面 `.permute({3,2,1,0}).unsqueeze(-1)`），`final_state_dlc = at::empty_like(initial_state_dlc)`
   同样是 5 维，但返回处写的是
   `return std::make_tuple(out_dlc.permute({3, 2, 1, 0}), final_state_dlc.permute({3, 2, 1, 0}));`
   —— 5 维张量用 4 阶 `permute`，抛出即失败。
   **建议修法**：两处各加 `.squeeze(-1)`，即 `out_dlc.squeeze(-1).permute({3,2,1,0})`、
   `final_state_dlc.squeeze(-1).permute({3,2,1,0})`（期望返回 `out=[1,N,H,Dv]`、`final_state=[B,H,Dv,Dk]`）。

2. **kernel 已经把设备卡死**：异常发生在 `kernel.launch()` **之后**，
   实测在失败调用后立刻做一次最普通的 `a+1.0` + `torch.dlc.synchronize()` 也会永久卡住
   （dlc:2、dlc:3 两台都被打到必须复位设备），说明 launch 出去的 kernel 本身也挂住了/越界。

3. **另需修复：`beta_logits` 被就地改写**。kernel 里
   `sigmoid_bf16(vmem_beta, vmem_beta, N, H); Vmem2HBM(vmem_beta, beta_hbm, ...)`
   （`dlc_kda_chunk_prefill.cpp:416-418`）把「sigmoid 结果」写回了**调用方的输入张量地址**，
   而 native_functions.yaml 里该参数没有 `Tensor(a!)` 标注（上层会当它只读）。
   在 CUDA graph / 多步复用下这是隐性数据污染，请改成写内部缓冲。
   注：kernel 内部对 beta 做 sigmoid、对 q（乘 `query_scale`）与 k 做 L2Norm(eps=1e-6) 是符合 GLM5 预期的。

4. **接口与 GLM5 的差异（请在修复时一并确认）**：
   - `g_bias` 公开签名是 `[D, H]`（wrapper 里 `g_bias.permute({1,0})`），
     而模型侧 `dt_bias` 是 `[H*D]` 扁平、原生实现 reshape 成 `[H, D]` —— 与同批的
     `dlc_kda_recurrent_decode`（要 `[H, D]`）**两个算子约定相反**，容易接错；
   - `initial_state` 必须是 4D `[B,H,Dv,Dk]` fp32（见下节「state 布局定稿」），而 wrapper 内部按 5 维处理（见第 1 点）；
   - `q/k/v/raw_g` 必须 `[1, N, H, 128]`（batch 维固定 1），`beta_logits` `[1,N,H]`。

**处置**：本条需求类型由「扩展已有算子能力」改为「**修复算子功能或精度**」，算子名称改回
`dlc_kda_chunk_prefill`；在修复完成（能正常返回且不卡设备）之前，插件侧继续使用
`vllm_cl/patch/patch_kda.py` 的逐 token 递推。修复后需重新跑一遍数值对齐
（reference：同一脚本里的 fp32 逐 token 递推，含逐通道 gate / bounded safe gate / L2Norm）。

## 优先级
P0

## 目标版本与期望时间
GLM-5.3-Flash 适配 Phase 2（性能阶段）。建议在长上下文（≥32K prefill）性能验收前提供，具体里程碑待算子组排期确认。

## 算子名称
修复 `torch.dlc_kda_chunk_prefill`（原提名算子已由算子组交付；当前 host wrapper 必抛异常 + kernel 卡死设备 + 就地改写输入 `beta_logits`）

（2026-09-22 之前的目标是「扩展 `torch.chunk_gated_delta_rule`」，现已由专用算子替代，仅当修复方案放弃时才回到扩展路线。）

## 背景与功能说明
使用场景：GLM-5.3-Flash 34 层 KDA 在 prefill 阶段的 chunked 递推，输入为短卷积后的 q/k/v、raw gate、beta，输出 core attention 结果与最终递归状态。
当前问题：插件里 `chunk_kda_with_fused_gate_dlc` **并没有真正的 chunk 实现**，而是直接复用逐 token 递推 `_run_kda`（`vllm_cl/patch/patch_kda.py:139`）。也就是说 prefill 的复杂度是 O(T) 的 Python 循环，每个 token 做：

```python
h = h * exp(gate)                                  # [H,Dv,Dk]
delta = (v - bmm(h, k)) * beta
h = h + delta[...,None] * k
out = bmm(h, q * scale)
```

单层 4096 token 需要 4096 次 Python 迭代 + 每次 2 个 bmm，34 层即约 28 万次迭代，长 prefill 完全不可用。
目标功能：真正的 chunked 并行实现（chunk 内矩阵化、chunk 间只传递状态），支持变长序列、gate 的 chunk 内累计衰减、首状态加载与末状态回写。
非目标：decode 单步递推（见 `dlc_kda_recurrent_decode`）；短卷积（见 01/02 号需求）。

## 期望 API 与函数签名
> **本条是「扩展已有算子」**：算子名保持 `torch.chunk_gated_delta_rule`，只是并入了逐通道 gate、
> bounded safe gate 与 chunk 内累计衰减；下面的签名是「新增项并入后的形态」，供算子组对照实现。

```python
def chunk_gated_delta_rule(  # 现有算子，仅新增参数/行为
    q: torch.Tensor,            # [1, T, H, Dk] bf16, 已 L2Norm（或由算子内做）
    k: torch.Tensor,            # [1, T, H, Dk] bf16
    v: torch.Tensor,            # [1, T, H, Dv] bf16
    raw_g: torch.Tensor,        # [1, T, H, Dk] bf16, 未过 sigmoid 的 gate logit
    beta: torch.Tensor,         # [1, T, H]     fp32, 已 sigmoid
    A_log: torch.Tensor,        # [H]           fp32
    g_bias: torch.Tensor,       # [H, Dk]       fp32
    initial_state: torch.Tensor,# [num_seqs, H, Dv, Dk] fp32（可为 gather 后的副本）
    cu_seqlens: torch.Tensor,   # [num_seqs + 1] int32
    safe_gate: bool = True,
    lower_bound: float = -5.0,
    scale: float | None = None, # 默认 Dk**-0.5
    use_qk_l2norm_in_kernel: bool = True,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # 返回 (core_out [1,T,H,Dv] bf16, final_state [num_seqs,H,Dv,Dk] fp32)
```

## 参数与返回值
输入：
- `q/k/v`：packed 形式 `[1, T, H, Dk]`，BF16；H 为 TP 分片后的头数（TP8 → 8），Dk=Dv=128。
- `raw_g`：`[1, T, H, Dk]`，BF16，未加 bias 的 gate logit。
- `beta`：`[1, T, H]`，FP32，**已 sigmoid 的写入门控**（区别于 decode 路径的 raw beta，见「支持范围」）。
- `A_log`：`[H]`，FP32；`g_bias`：`[H, Dk]`，FP32（公开签名按 wrapper 惯例收 `[Dk,H]`，内部 `permute({1,0})`）。
- `initial_state`：`[num_seqs, H, Dv, Dk]`，FP32，变长序列各自的首状态。
- `cu_seqlens`：`[num_seqs+1]`，int32，单调不减，首元素 0。

属性：
- `safe_gate`：是否在 kernel 内计算 `gate = lower_bound * sigmoid(exp(A_log) * (raw_g + g_bias))`。
- `lower_bound`：默认 `-5.0`（GLM-5.3-Flash `gate_lower_bound`）。
- `scale`：输出缩放，默认 `Dk**-0.5`。
- `use_qk_l2norm_in_kernel`：是否在 kernel 内对 q/k 做 L2Norm（reference 中是 `q*rsqrt(sum(q^2)+1e-6)`）。

返回值：
- `core_out`：`[1, T, H, Dv]`，BF16，KDA read 结果（未经 gated RMSNorm）。
- `final_state`：`[num_seqs, H, Dv, Dk]`，FP32；`output_final_state=False` 时为 None。
- 不做 in-place 写回，状态写回由调用方 `scatter_states` 负责（可与本算子可选融合）。

## 功能语义
论文记法 `S ∈ R^(Dk×Dv)`，每个 head 独立。chunk 大小记为 C。

单 token 递推（等价参考）：

```text
g_t     = lower_bound * sigmoid(exp(A_h) * (raw_g_t + g_bias_h))   # safe_gate
alpha_t = exp(g_t)                                                 # [Dv]
S       = diag(alpha_t) @ S
v_hat   = S^T @ k_t
delta_v = beta_t * (v_t - v_hat)
S       = S + k_t @ delta_v^T
o_t     = (S^T @ q_t) * scale
```

chunked 实现：

```text
chunk 内：
  - gate 前缀累计（累计 log 衰减）→ 与论文的 chunked delta rule 一致
  - 用 causal mask 的矩阵形式一次性算出 chunk 内贡献
chunk 间：
  - 传入/传出 S，只有状态是串行的
```

broadcasting / padding：变长序列由 `cu_seqlens` 切分，chunk 不跨序列边界；序列长度不足一个 chunk 时按实际长度处理。
in-place / alias：不修改输入；`initial_state` 只读。
空 / 零长度：`T == 0` 或空序列返回空输出并保持状态不变。
异常输入：`cu_seqlens` 非单调 / 越界、`beta` 形状不符、`A_log` 长度不等报错。
确定性：无数值随机性；同一输入结果确定。
累加精度：状态与 gate 全程 FP32，输出一次性 cast 到 BF16。

## 支持范围与约束
支持的 dtype：`q/k/v/raw_g` BF16；`beta/A_log/g_bias/initial_state` FP32。若算子组要求全 BF16，请明确说明并给出精度影响。
支持的 shape：`T` 覆盖 1~32768；`H` 动态（TP 分片：8 / 16 / 32）；`Dk=Dv=128`（GLM-5.3-Flash 固定，但建议不写死）。`num_seqs` 动态。
数值范围：`alpha = exp(gate)` 而 `gate ∈ [-5, 0)`，故 `alpha ∈ (0.0067, 1]`，**不会溢出**——这是选择 chunked 矩阵化的数值前提，请勿依赖更宽的 gate 范围。
layout：输入连续；packed 第一维为 1。
device：DLC 单卡。
执行模式：prefill 走 eager 即可，但需可被 CUDA graph 捕获（可选）。
其他限制：**beta 已经 sigmoid**（与 decode 路径的 `sigmoid_beta=True` 相反），接口需显式区分，避免重复 sigmoid；MTP 未启用时 `T` 无 num_spec 附加维。

## 精度要求
Golden / reference：`vllm_cl/patch/patch_kda.py:93` 的 `_run_kda`（逐 token FP32 参考），在 CPU/DLC 上跑短序列（T ≤ 512）作为 golden。
输入分布：`q/k/v ~ N(0,1)`；`raw_g` 均匀覆盖 `[-8, 2]`（保证 gate 落在边界附近）；`beta ∈ (0,1)`；种子固定。
样本数量：每个长度档 ≥ 100 组。
比较 dtype：BF16 输出对 FP32 reference。
atol / rtol：`core_out atol=2e-2, rtol=2e-2`；`final_state atol=1e-3, rtol=1e-3`（FP32 状态，要求更严）。
必须逐元素一致：`T=1` 且 `raw_g`/`beta` 为精确可表示值时，要求与 decode 单步 kernel 输出 bit-exact（两条路径必须一致）。
NaN / Inf / 极值：`gate` 取 `lower_bound` 边界、`alpha` 最小 0.0067 时不失稳。
累加与中间精度：state / gate / softmax 类中间量 FP32。

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
1. 基本功能：单序列 T ∈ {1, 64, 128, 256, 512, 1024, 4096}；多序列变长 [1, 300, 1024, 4096]。
2. chunk 边界：T 恰为 chunk 整数倍 / 余 1 / 少 1，结果一致。
3. 与 decode 路径一致性：同输入同一 token 位置，chunk kernel 与 `dlc_kda_recurrent_decode` 输出一致。
4. 状态：`initial_state` 非零 / 全零；`output_final_state=False` 时不返回状态且不报错。
5. 动态 shape：`H ∈ {8, 16, 32}`，`Dk ∈ {128}`（可加 64/256 泛化）。
6. 异常：`cu_seqlens` 非法、形状不匹配。
7. 性能：T=4096、34 层、TP8 下单次 prefill 耗时；目标相对逐 token native 实现 ≥ 20×，并给出 MFU/带宽口径。

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
Reference 实现：`vllm_cl/patch/patch_kda.py:93`（`_run_kda`）、`:139`（`chunk_kda_with_fused_gate_dlc`）。
调用点：`vllm/models/glm5next/nvidia/kda.py:545`（prefill 分支）。
上游语义参考：`vllm/third_party/flash_linear_attention/ops/kda.py` 的 `chunk_kda_with_fused_gate`（CUDA/Triton 版）。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md` §5.2/§5.3/§5.4。
