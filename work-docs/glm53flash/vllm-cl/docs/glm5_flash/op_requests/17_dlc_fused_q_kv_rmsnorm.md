<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22）：可复用已有算子（本条撤回，不提交） -->
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：语义已被覆盖（建议撤回或降为 P2）
     证据：dlc_fused_rms_norm_f32_bf16 对 2D packed 行归一化实测 err 6.02e-04，Q/KV 各调用一次即可等价（输入需 fp32、weight bf16、输出 fp32）
     建议：撤回；若坚持单 launch 融合再降为 P2 提交 -->

# [Operator Request] dlc_fused_q_kv_rmsnorm（视觉塔 Q/K RMSNorm）

## 需求类型
其他

**需求分类：可复用已有算子（本条撤回，不提交）**
- 现有算子：`torch.dlc_fused_rms_norm_f32_bf16(input, weight, eps)`（要求 `input` fp32、`weight` 为 1D bf16；输出 fp32）。
- 实测（dlc:0；`[128, 8192]` 的 packed 行归一化）：`err=6.02e-04`，与 reference 一致 → 对 Q、KV 各调用一次即可等价覆盖 `fused_q_kv_rmsnorm`。
- 结论：语义已覆盖，剩余价值只是「两次 launch 合成一次」；建议撤回，或在多模态（Phase 3）阶段作为可选优化重新评估。
- 复现脚本：`probes/probe_fused_rms_norm_packed_rows.py`。
- 本次调整（2026-09-22）：需求类型「优化算子性能」→「其他（复用已有算子）」；优先级 P2 不变；算子名称改为复用 `torch.dlc_fused_rms_norm_f32_bf16`（调用两次）；**本条撤回，不提交**。

## 优先级
P2

## 目标版本与期望时间
本条不提交给算子组、不需要排期：插件侧在 Phase 3（多模态）前把视觉塔 Q/K RMSNorm 换成两次 `torch.dlc_fused_rms_norm_f32_bf16` 即可（算子组无需改动）。
## 算子名称
复用 `torch.dlc_fused_rms_norm_f32_bf16`（Q 与 KV 各调用一次；原提名 `dlc_fused_q_kv_rmsnorm` 撤回）

## 背景与功能说明
使用场景：GLM-5.3-Flash 视觉塔（24 层）在每个 attention block 里对 q 和 kv 分别做 RMSNorm，24 层 × 2 个张量。
当前问题：上游是 Triton kernel（`vllm/models/common/ops/fused_qk_rmsnorm.py`），DLC 无法执行；在 profile 阶段的多模态 dummy 前向上直接报错。插件已用纯 PyTorch 替换（`vllm_cl/patch/patch_fused_qk_rmsnorm.py`）：

```python
xf = x.float()
rrms = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
y = (xf * rrms * weight.float()).to(x.dtype)
```

即每个张量 4~5 个 elementwise op + FP32 中间张量。
目标功能：单 kernel 完成 q 与 kv 两个张量的 FP32 RMSNorm。
非目标：文本侧 MLA 的 `fused_rms_norm_qkv_rotary_emb`（那是另一个已存在的 DLC 算子）。

## 期望 API 与函数签名
> **本条已撤回（不提交）**：下面这段签名是初稿拟定的形式，仅作存档；实际动作是复用现有算子 `torch.dlc_fused_rms_norm_f32_bf16`（Q / KV 各调一次）
> （见「需求分类」段与「算子名称」），**不需要算子组提供这个新 API**。

```python
def fused_q_kv_rmsnorm(
    qr: torch.Tensor,        # [T, q_dim] bf16
    kv: torch.Tensor,        # [T, kv_dim] bf16
    q_weight: torch.Tensor,  # [q_dim] 与 qr 同 dtype
    kv_weight: torch.Tensor, # [kv_dim]
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 返回 (qr_out, kv_out)，均与各自输入同形同 dtype
```

## 参数与返回值
输入：
- `qr` / `kv`：`[T, q_dim]` / `[T, kv_dim]`，BF16，2D；两者 `T` 相同。
- `q_weight` / `kv_weight`：对应宽度的一维权重。
- `eps`：视觉塔为 1e-5。

返回值：
- `(qr_out, kv_out)`：与输入同形同 dtype；不修改输入。

边界：`T == 0` 返回空张量（仍需返回正确形状）。
异常：`qr.shape[0] != kv.shape[0]`、权重长度不匹配报错。

## 功能语义
```text
y = x.float() * rsqrt(mean(x.float()^2, dim=-1, keepdim=True) + eps) * weight.float()
out = y.to(x.dtype)
```

q 与 kv 各自独立归约；两路可并行。
in-place / alias：不修改输入（上游实现使用独立输出缓冲）。
空 / 零长度：`T == 0` 返回空。
异常：形状不匹配报错。
确定性：无随机。
累加精度：平方和归约 FP32，输出一次性 cast。

## 支持范围与约束
支持的 dtype：BF16（建议同时支持 FP16）。
支持的 shape：`T` 动态（图像 token 数）；`q_dim`/`kv_dim` 动态（GLM-5.3-Flash 视觉塔 hidden=1024、heads=16）。
数值范围：常规 BF16；`mean(x^2)` 接近 0 时靠 `eps` 保稳。
layout：2D 连续，最后一维连续。
device：DLC 单卡。
执行模式：多模态 prefill 走 eager 即可。
其他限制：需保持与上游 Triton 版完全相同的数值口径（FP32 归约、单次 cast、空输入返回空缓冲）。

## 精度要求
Golden / reference：`vllm_cl/patch/patch_fused_qk_rmsnorm.py:13`（`_rms_norm_fp32`）；上游 Triton 版 `vllm/models/common/ops/fused_qk_rmsnorm.py` 可作交叉参考。
输入分布：`qr/kv ~ N(0,1)`，权重 `~ U(0.5,1.5)`；含全 0 输入。
样本数量：≥ 200 组。
比较 dtype：BF16 输出对 FP32 reference。
atol / rtol：`atol=1e-2, rtol=1e-2`。
必须逐元素一致：`T=0` 时必须返回正确形状的空张量（不可返回 None 或形状错误）。
NaN / Inf / 极值：全 0 输入输出为 0（`rsqrt(eps)` 有限）。
累加精度：FP32。

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
1. 功能：T ∈ {1, 64, 4096}，`q_dim`/`kv_dim` 取视觉塔实际值。
2. 边界：T=0；`qr` 与 `kv` 宽度不同。
3. 精度 / layout / 异常。
4. 端到端：与 `patch_fused_qk_rmsnorm.py` 替换后的多模态 profile 前向对比，输出一致。
5. 性能：T=4096 下单次耗时；对比 native 组合（目标 ≥ 2×）。

## 验证环境与依赖版本
DLC 硬件：8 卡，设备名 `/dev/cltech0-7`（TP=8，无 PP）
vLLM commit：`f5361fb09f`
vLLM DLC（vllm-cl）commit：`a14fc5b`
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：`VLLM_USE_DLC_COL_MAJOR_MATMUL=1`
模型：GLM-5.3-Flash-Mix（AWQ 4-bit，`linear_num_heads=64`、`linear_head_dim=128`、`short_conv_kernel_size=4`、`gate_lower_bound=-5.0`）

## 验收标准
- [ ] 视觉塔对 Q、KV 各调用一次 `dlc_fused_rms_norm_f32_bf16`（`input` fp32、`weight` bf16、输出 fp32，再 cast 回 bf16）。
- [ ] 与 native reference 一致（实测 `err=6.02e-04`）。
- [ ] 接入后多模态 profile 前向不再因缺 `fused_qk_rmsnorm` 报错。
- [ ] 不修改现有算子：本条不产生算子组接口/行为变更。
## Reference、日志与附件
Reference 实现：`vllm_cl/patch/patch_fused_qk_rmsnorm.py`（DLC native 替换）。
上游实现：`vllm/models/common/ops/fused_qk_rmsnorm.py`（Triton）。
调用点：`vllm/models/glm5next/nvidia/multimodal.py`；同时被 DeepSeek-V4 / Kimi-K3 复用。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_联调问题与修复记录.md` 问题 1。
