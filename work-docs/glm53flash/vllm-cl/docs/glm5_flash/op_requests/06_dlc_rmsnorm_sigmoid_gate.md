<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22 定稿）：完全新算子 -->
<!-- 需求分类（2026-09-22 二次，算子组交付新算子后实测）：可复用（撤回，不提交）
     算子组已交付 torch.dlc_kda_rmsnorm_sigmoid_gate（kernel custom_kda_rmsnorm_sigmoid_gate_bf16），
     实测 sigmoid 语义正确（err 0.34%），3D/4D、weight bf16/fp32、T=0 均通过 → 本条撤回，转插件接入。 -->
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：未覆盖（保留新增）
     证据：DLC 只有 dlc_fused_rms_norm_f32_bf16（无 gate，且 fp32 入/fp32 出）与 dlc_fused_rms_norm_qkv_rotary_emb（带 rope），kernel 清单亦无 gate 变体
     建议：保留新增；若算子组倾向复用，可改为给 dlc_fused_rms_norm_f32_bf16 增加 gate 输入与 activation 参数 -->

# [Operator Request] dlc_rmsnorm_sigmoid_gate（KDA 输出 gated RMSNorm）

## 需求类型
其他

**需求分类：完全新算子**
- 核对结果（2026-09-21）：DLC 现有 RMSNorm 系列只有 `dlc_fused_rms_norm_f32_bf16`（无 gate；且要求 fp32 输入、输出 fp32）与 `dlc_fused_rms_norm_qkv_rotary_emb`（融合的是 rotary 而非 gate）；`DLC_Custom_Kernel/dlc_kernels/norm/` 下也没有 gate 变体。
- 因此本需求保持「新增算子」。若算子组希望减少算子数量，可接受的最小改动形态是给 `dlc_fused_rms_norm_f32_bf16` 增加 `gate`（bf16）与 `activation`（`sigmoid`/`silu`）参数，等价于上游 `FusedRMSNormGated`。
- 本次调整（2026-09-22）：需求类型与优先级（P1）均不变；覆盖核查确认无可用现有算子，并给出算子组可接受的替代形态。
- 本次调整（2026-09-22 二次，算子组交付新算子后实测）：算子组已交付 `dlc_kda_rmsnorm_sigmoid_gate` 并实测满足
   （sigmoid 版 err 0.34%，3D/4D、bf16/fp32 weight、T=0 边界均通过），故**本条撤回、不再提交**；
   需求类型改为「其他」，优先级 P1→P2，动作转为插件侧接入。


## 2026-09-22 新算子实测（算子组已交付 `dlc_kda_rmsnorm_sigmoid_gate`）——本条撤回

算子组已提交专用算子 `torch.dlc_kda_rmsnorm_sigmoid_gate(core_out, gate, norm_weight, eps=1e-5)`
（`aten/src/ATen/native/dlc/KDARMSNormSigmoidGate.cpp`，kernel `custom_kda_rmsnorm_sigmoid_gate_bf16`），
实测完全满足本条需求，故**本需求撤回、不再提交**，改为插件侧接入动作。

实测脚本：`probes/probe_new_kda_rmsnorm_gate.py`（对照 `patch_kda.FusedRMSNormGatedDLC` 的 fp32 reference）。

| 用例 | sigmoid 版 err | swish 版 err | 结论 |
|---|---|---|---|
| 4D `[1,64,8,128]`，weight bf16 | **1.56e-02**（\|ref\|max 4.53 → 0.34%） | 1.50e+01 | **实现的是 sigmoid 版**，与 GLM5 `activation="sigmoid"` 一致 |
| 3D `[16,8,128]` | 3.13e-02 | 2.07e+01 | 3D/4D 均支持 |
| weight fp32 | 1.56e-02 | 1.20e+01 | bf16/fp32 weight 均可 |
| T=1 | 7.81e-03 | 1.01e+01 | 通过 |
| T=0 | 不报错，返回同形空张量 | — | 通过 |
| `D=64` / `x` 与 `g` 形状不一致 | 报错 `core_out hidden dimension must be 128` / `shape must match` | — | 校验清晰 |

**接入要点**：
- 固定 `hidden==128`（GLM5 的 `head_dim=128` 正好满足）；
- `core_out`/`gate` 最后一维必须连续（`stride(-1)==1`），GLM5 的 `core_attn_out`/`g2` 天然满足；
- 语义 = `y = x * rsqrt(mean(x²)+eps) * w * sigmoid(g)`（FP32 归约，输出一次 cast 回 bf16），
  与 `patch_kda.FusedRMSNormGatedDLC(activation="sigmoid")` 等价（残差 0.34% 来自输出 bf16 舍入）；
- 插件接入点：`vllm_cl/patch/patch_kda.py:145` `FusedRMSNormGatedDLC.forward`
  （调用点 `vllm/models/glm5next/nvidia/kda.py:261/335`），替换后需重跑 34 层端到端数值对齐；
- 撤回后本条的优先级降为 P2（不再是算子组工作项，仅为插件接入清单项）。

## 优先级
P2

## 目标版本与期望时间
本条不提交给算子组、不需要排期：算子组已交付 `torch.dlc_kda_rmsnorm_sigmoid_gate` 并实测满足，
插件侧在 Phase 2 内完成接入即可（34 层 × 每 token 一次的高频小算子，接入后按端到端耗时验收）。

## 算子名称
`torch.dlc_kda_rmsnorm_sigmoid_gate`（算子组实际交付名；上游名 `FusedRMSNormGated`；初稿曾写 `torch.dlc_rmsnorm_sigmoid_gate`）

## 背景与功能说明
使用场景：KDA 每层在 read 得到 `core_out` 后，要按 head 做 RMSNorm，再用 `g_out` 做 sigmoid 门控，最后 flatten 送 `o_proj`。
当前问题：native `FusedRMSNormGatedDLC`（`vllm_cl/patch/patch_kda.py:145`）由 5~6 个 elementwise 算子组成：

```python
xf = x.float()
y  = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
y  = y * weight.float()
y  = y * g.float().sigmoid()
return y.to(x.dtype)
```

每次涉及 FP32 中间张量 `[T,H,128]` 的多次读写。34 层 × 每 token 一次，属于高频小算子。
目标功能：单 kernel 完成「按最后一维 RMSNorm（FP32 归约）→ 乘 norm weight → 乘 sigmoid(gate) → cast 回 BF16」。
非目标：mHC 的 RMSNorm（见 mHC 需求）；不需要支持 residual/prenorm（GLM-5.3-Flash 不传递这两个参数）。

## 期望 API 与函数签名
```python
def fused_rmsnorm_sigmoid_gate(
    x: torch.Tensor,        # [T, H, D] 或 [1, T, H, D] bf16
    g: torch.Tensor,        # 与 x 同形 bf16（门控输入，未过 sigmoid）
    weight: torch.Tensor,   # [D] bf16/fp32，RMSNorm 权重
    eps: float,             # 例如 1e-5
    activation: str = "sigmoid",
) -> torch.Tensor:          # 与 x 同形 bf16
```

## 参数与返回值
输入：
- `x`：`[T, H, D]`（GLM-5 实际为 3D，H 在 `shape[-2]`），BF16。
- `g`：与 `x` 同形，BF16，`g_out` 门控。
- `weight`：`[D]`，RMSNorm 权重。
- `eps`：归约稳定项，GLM-5.3-Flash 为 1e-5。

属性：
- `activation`：默认 `"sigmoid"`（GLM-5.3-Flash 使用）；若算子组希望兼容上游 `swish`，可一并声明。

返回值：
- `y`：与 `x` 同形同 dtype，BF16。

边界：`T == 0` 返回空；`weight=None` 时只做 RMSNorm + 门控（可选）。
异常：`x` 与 `g` 形状不一致报错。

## 功能语义
```text
xf = x.float()
r  = rsqrt(mean(xf^2, dim=-1, keepdim=True) + eps)
y  = xf * r * weight.float() * sigmoid(g.float())
return y.to(bf16)
```

broadcasting：`weight` 沿最后一维广播。
in-place / alias：不修改输入；可支持 out 缓冲（可选）。
空 / 零长度：`T == 0` 返回空。
异常：形状不匹配报错。
确定性：无随机。
累加精度：平方和归约用 FP32，输出一次性 cast。

## 支持范围与约束
支持的 dtype：`x/g` BF16，`weight` BF16 或 FP32（请算子组确认），内部 FP32。
支持的 shape：`T` 动态；`H` 为 TP 分片头数（8 @TP8）；`D=128`（建议支持 64~256）。
数值范围：`x` 常规 BF16；`mean(x^2)` 接近 0 时靠 `eps` 保稳。
layout：最后一维连续。
device：DLC 单卡。
执行模式：需支持 CUDA graph。

## 精度要求
Golden / reference：`vllm_cl/patch/patch_kda.py:145`（`FusedRMSNormGatedDLC.forward`）。
输入分布：`x ~ N(0,1)`，`g` 覆盖 `[-10, 10]` 以覆盖 sigmoid 两端；另加全 0 输入。
样本数量：≥ 200 组。
比较 dtype：BF16 输出对 FP32 reference。
atol / rtol：`atol=1e-2, rtol=1e-2`。
必须逐元素一致：不要求（存在 FP32 中间量与 cast）。
NaN / Inf / 极值：全 0 输入时输出为 0（`rsqrt(eps)` 有限）；`g` 极大时 sigmoid 饱和为 1，不产生 NaN。
累加精度：平方和 FP32。

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
1. 功能：T ∈ {1, 64, 4096}，H ∈ {8, 16}，D=128。
2. 边界：T=0、全 0 输入、`g` 饱和。
3. 精度：`atol/rtol` 达标。
4. layout / 异常 / CUDA graph。
5. 性能：对比 native 组合（预期 ≥ 3× 带宽收益），报告每层耗时。

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
Reference 实现：`vllm_cl/patch/patch_kda.py:145`（`FusedRMSNormGatedDLC`）。
调用点：`vllm/models/glm5next/nvidia/kda.py:261`（`self.o_norm = FusedRMSNormGated(head_dim, activation="sigmoid")`）。
上游语义参考：`vllm/third_party/flash_linear_attention/ops/kda.py` 的 `FusedRMSNormGated`。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md` §5.1（`g_out` → gated RMSNorm）。
