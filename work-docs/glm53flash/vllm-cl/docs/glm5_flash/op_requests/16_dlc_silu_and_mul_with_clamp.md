<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22 二次复核）：新增独立算子（原判「扩展 silu_and_mul」，复核后改为「新增」） -->
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：未覆盖
     证据：DLC 只有 silu_and_mul / silu_and_mul_with_padding，torch.silu_and_mul_with_clamp 在 DLC torch 中不存在（hasattr 探测 MISSING）
     复核（2026-09-22）：不改共享的 silu_and_mul 签名，改为新增独立算子，与上游 _C.silu_and_mul_with_clamp 同名同语义
     理由：1) 上游本身就是独立算子（activation.py:235 分派 torch.ops._C.silu_and_mul_with_clamp），silu_and_mul 不带 clamp 参数；
           2) silu_and_mul 在所有模型的 FFN 热路径上（vllm_cl/ops/activation.py 的 DLCSiluAndMul + _custom_ops.py:1165），
              且上游有基于 torch.ops._C.silu_and_mul.default 的量化融合 pattern（act_quant_fusion.py:31）与 .out 变体，
              加参数会波及不相关模型与融合 pass；
           3) clamp 语义是非对称变体（gate 只 clamp 上界、up 双侧 clamp + alpha/beta），不是同一算子的模式开关；
           4) DLC 已有「一个 kernel 源码、多个 ATen op」的先例（silu_and_mul 与 silu_and_mul_with_padding 共用 custom_silu_and_mul）。
     注：routed expert 侧不需要本算子——dlc_vllm_fused_moe_w4a16_awq_col_major 已有 swiglu_limit 参数（vllm_cl/ops/awq_marlin.py:366-407）。 -->

# [Operator Request] dlc_silu_and_mul_with_clamp（带 clamp 的 SwiGLU 激活）

## 需求类型
新增算子

**需求分类：完全新算子（新增独立算子 `dlc_silu_and_mul_with_clamp`，不改 `silu_and_mul` 签名）**
- 现有算子：`torch.silu_and_mul` / `silu_and_mul.out`（DLC 已注册，`native_functions.yaml:6086`）与 `silu_and_mul_with_padding`；**没有 clamp 变体**（`probes/probe_op_surface.py` 实测 `hasattr(torch, 'silu_and_mul_with_clamp') == False`）。
- 需求语义（GLM5 `swiglu_limit=10.0`，所有非 MoE 前馈路径）：`gate = clamp(x[..., :d], max=limit)`、`up = clamp(x[..., d:], min=-limit, max=limit)`、`out = gate * sigmoid(alpha * gate) * (up + beta)`。
- **复核结论（2026-09-22）：新增独立算子，而不是给 `silu_and_mul` 加参数**：
  1. **上游就是独立算子**：`vllm/model_executor/layers/activation.py:235` 在 CUDA-alike 上直接分派 `torch.ops._C.silu_and_mul_with_clamp`；上游的 `silu_and_mul` 始终只有 `(out, x)` 两个参数。
  2. **`silu_and_mul` 是所有模型共享的 FFN 热路径**：`vllm_cl/ops/activation.py:14` 把 `DLCSiluAndMul` 注册为 `SiluAndMul` 的 OOT 实现，任何模型的 dense、shared-expert、视觉 FFN 都会调它；上游还有 `act_quant_fusion.py:31` 基于 `torch.ops._C.silu_and_mul.default` 的量化融合 pattern 与 `.out` 变体引用同一 op。给它加参数会牵动不相关模型和融合 pass。
  3. **语义上是另一个激活变体**：clamp 是非对称的（gate 只 clamp 上界，up 双侧 clamp）并带 `alpha`/`beta`（SwiGLU-OAI），不是「同一算子的可选开关」；现有调用方不存在「需要带 clamp 的 silu_and_mul」这种场景。
  4. **不影响复用 kernel 实现**：DLC 已有「一份 kernel 源码注册多个 ATen op」的先例（`silu_and_mul` 与 `silu_and_mul_with_padding` 共用 `custom_silu_and_mul(_bf16)`，`UnaryOpsKernel.cxx:979/1011`），新 op 可以在同一份 kernel 上实现/扩展，只是**不改旧 op 的签名**。
- 复现脚本：`probes/probe_op_surface.py`。
- 本次调整（2026-09-22）：需求类型「扩展已有算子能力」→「新增算子」；优先级 P1 不变；算子名称定为**新增** `torch.dlc_silu_and_mul_with_clamp`（不再改 `silu_and_mul` 签名）。

## 优先级
P1

## 目标版本与期望时间
GLM-5.3-Flash 适配 Phase 2（性能阶段）。该算子影响所有前馈路径，建议与 KDA 算子并行推进。

## 算子名称
新增 `torch.dlc_silu_and_mul_with_clamp`（与上游 `torch.ops._C.silu_and_mul_with_clamp` 同名同语义；不改 `silu_and_mul` 的签名）

## 背景与功能说明
使用场景：GLM-5.3-Flash 的**所有**前馈路径都要求带 clamp 的 SwiGLU：dense FFN（前 3 层）、shared expert、routed expert（fused MoE 内部已由 DLC kernel 的 `swiglustep` 处理）、以及视觉塔 MLP，`swiglu_limit=10.0`。
当前问题：vLLM 的 `SiluAndMulWithClamp`（`vllm/model_executor/layers/activation.py:204`）是 `CustomOp`，DLC 只注册了不带 clamp 的 `DLCSiluAndMul`（`vllm_cl/ops/activation.py:14`），因此 `SiluAndMulWithClamp` 在 DLC 上落回 `forward_native`：

```python
gate = torch.clamp(x[..., :d], max=self.swiglu_limit)
up   = torch.clamp(x[..., d:], min=-self.swiglu_limit, max=self.swiglu_limit)
return gate * torch.sigmoid(self.alpha * gate) * (up + self.beta)
```

即 4~5 个 elementwise op + 中间张量读写，且 `x` 为 `[T, 2*intermediate]`（dense 层 intermediate=12288，视觉 4096），带宽开销明显。
目标功能：单 kernel 完成 clamp(gate) / clamp(up) / SiLU / 乘，并保留上游的 `alpha`/`beta` 泛化（SwiGLU-OAI 风格）。
非目标：fused MoE 内部的 expert 激活（已由 `swiglustep` 覆盖）；router 的 TopK。

## 期望 API 与函数签名
```python
def silu_and_mul_with_clamp(
    x: torch.Tensor,          # [T, 2*d] 或 [B, S, 2*d] bf16
    swiglu_limit: float,      # GLM-5.3-Flash = 10.0
    alpha: float = 1.0,
    beta: float = 0.0,
    out: torch.Tensor | None = None,   # [T, d]
) -> torch.Tensor:            # [T, d] bf16
```

## 参数与返回值
输入：
- `x`：`[T, 2*d]`（或 `[B, S, 2*d]`），BF16；前半为 gate，后半为 up。
- `swiglu_limit`：float，GLM-5.3-Flash 为 10.0；需支持任意非负值。
- `alpha` / `beta`：float，默认 1.0 / 0.0（GLM-5.3-Flash 用默认值）。

返回值：
- `out`：形状 `x.shape[:-1] + (d,)`，BF16；提供 `out` 参数时原位写并返回。

边界：`T == 0` 返回空；`d = x.shape[-1] // 2` 必须为正整数。

## 功能语义
```text
d    = x.shape[-1] // 2
gate = clamp(x[..., :d],      max=swiglu_limit)
up   = clamp(x[..., d:], min=-swiglu_limit, max=swiglu_limit)
out  = gate * sigmoid(alpha * gate) * (up + beta)
```

说明：`alpha=1, beta=0` 时退化为 `silu(gate) * up`；`beta=1` 时为 SwiGLU-OAI。
in-place / alias：不修改 `x`；`out` 可原位写。
空 / 零长度：`T == 0` 返回空。
异常：`x.shape[-1]` 为奇数或 0 报错；`swiglu_limit < 0` 报错。
确定性：无随机。
累加精度：elementwise，无归约，BF16 输入输出即可（`sigmoid` 内部按 FP32 计算更稳，请算子组确认口径）。

## 支持范围与约束
支持的 dtype：输入输出 BF16（建议同时支持 FP16/FP32 输入）。
支持的 shape：`T` 动态（1~32K）；`d` 动态（GLM-5.3-Flash：dense 12288 / 视觉 4096 / MoE expert 2048）；支持 3D 输入 `[B,S,2d]`。
数值范围：`gate` 可达 ±1e4（需 clamp 后稳定）；`sigmoid` 饱和不产生 NaN。
layout：最后一维连续；优先支持 `[T, 2d]` 连续布局。
device：DLC 单卡。
执行模式：prefill 与 decode 都需要，decode 需支持 CUDA graph。
其他限制：
1. **不改动 `silu_and_mul` / `silu_and_mul.out` / `silu_and_mul_with_padding` 的签名与行为**：无 clamp 的模型继续走原 op，新算子只服务「模型配置带 swiglu_limit」的调用方。
2. 可以在同一份 kernel 源码上实现（`custom_silu_and_mul(_bf16)` 已有被两个 op 共用的先例），但必须注册成新 op；`swiglu_limit` 为必填参数（上游同款语义，不带「None 跳过 clamp」的模式）。
3. 现有 `silu_and_mul` 的无 clamp 路径需回归通过（新 op 不得影响旧 op 的编译/分派）。

## 精度要求
Golden / reference：`SiluAndMulWithClamp.forward_native`（`vllm/model_executor/layers/activation.py:238`）。
输入分布：`x ~ N(0,1)`；另加超过 ±10 的极端值以验证 clamp 生效；定向覆盖 gate 恰为 ±10 的边界。
样本数量：≥ 200 组。
比较 dtype：BF16 输出对 FP32 reference。
atol / rtol：`atol=1e-2, rtol=1e-2`。
必须逐元素一致：clamp 边界值（gate=10、up=±10）处输出必须与 reference 一致；同时核对新 op 上线后原 `silu_and_mul` 的输出不变（bit-exact）。
NaN / Inf / 极值：`x=±Inf` 经 clamp 后不得为 NaN；`x` 全 0 时输出 0。
累加精度：无归约；`sigmoid` 建议 FP32。

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
1. 功能：T ∈ {1, 64, 4096}，d ∈ {2048, 4096, 12288}；`alpha=1,beta=0` 与 `alpha,beta≠默认` 两组。
2. clamp 定向：gate/up 取 ±10、±10±1e-3 的边界样本。
3. 回归：原 `silu_and_mul` / `silu_and_mul_with_padding` 的输出与耗时不变（新 op 不改变旧 op 的签名与分派）。
4. 3D 输入 `[B,S,2d]`。
5. 边界（T=0、d 非法）/ 异常 / CUDA graph。
6. 性能：T=4096、d=12288 下带宽与耗时；对比 native 多算子（目标 ≥ 2×）。

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
Reference 实现：`vllm/model_executor/layers/activation.py:238`（`SiluAndMulWithClamp.forward_native`）。
现有 DLC 实现：`vllm_cl/ops/activation.py:14`（`DLCSiluAndMul`）、`vllm_cl/_custom_ops.py:1165`（`silu_and_mul`）。
调用点：`vllm/models/glm5next/nvidia/model.py:140`、`vllm/models/glm5next/nvidia/multimodal.py:106/324`。
MoE 侧已由 `swiglustep` 覆盖：`vllm_cl/ops/awq_marlin.py:370`。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md` §8.1（SwiGLU clamp 的隐藏正确性问题）。
