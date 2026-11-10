<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22，二次复核维持）：需扩展已有算子能力（不新增算子）
     复核依据：norm_weight/norm_eps 是上游 MHCPreOp 在所有后端都已存在的可选参数（norm_weight=None 时不做 norm），
     故属「补齐签名 / 对齐上游」，不是给旧算子加 GLM5 专属分支；DLC 侧调用方仅 GLM5 系列 mHC。 -->
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：已覆盖，但缺子层 RMSNorm（建议改为「扩展已有算子能力」）
     证据：dlc_mhc_pre 实测与 reference 一致（post 3.03e-05 / cm 1.06e-04 / layer_input 3.92e-03），Sinkhorn 已在 kernel 内；但签名无 norm_weight/norm_eps，插件只能在外层补 RMSNorm（多一次读写）；另注意返回的 cm 为 flat [T, hc*hc]
     建议：改为扩展 dlc_mhc_pre：增加 norm_weight/norm_eps，在 kernel 内对 layer_input 做 RMSNorm -->

# [Operator Request] dlc_mhc_pre（mHC 前置四路残差混合）

## 需求类型
扩展已有算子能力

**需求分类：需扩展已有算子能力（不新增算子）**
- 现有算子：`torch.dlc_mhc_pre(residual, fn, hc_scale, hc_base, rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=1.0, sinkhorn_repeat=1, n_splits=1) -> (post_mix, cm, layer_input)`（实现：`aten/src/ATen/native/dlc/mhc_pre.cxx`，**Sinkhorn 迭代已在 kernel 内**）。
- 实测（dlc:0；N=64 / hc_mult=4 / hidden=256）：与 reference 一致 —— `post_err=3.03e-05`、`cm_err=1.06e-04`（`cm` 为 flat `[N, hc*hc]`）、`layer_input_err=3.92e-03`（bf16 输出）。
- 唯一缺口：签名没有 `norm_weight` / `norm_eps`，而上游 `MHCPreOp` 会在 `layer_input` 上再做一次子层 RMSNorm（`_apply_mhc_norm`）。当前插件只能在外层用 `forward_oot` 补一次，多一轮 HBM 往返（`[N, hidden]` 读+写）。
- 期望扩展：给 `dlc_mhc_pre` 增加 `norm_weight: Tensor?` / `norm_eps: float`，在 kernel 内对 `layer_input` 做 RMSNorm（`norm_weight=None` 时行为与现状完全一致，保证向后兼容）。
- **为什么「扩展」而不是「新增算子」（2026-09-22 复核）**：`norm_weight`/`norm_eps` 不是 GLM5 新造的语义，而是上游 `MHCPreOp` 在**所有后端都已存在的可选参数**——
  `forward_cuda/hip/xpu` 的签名里都有 `norm_weight=None, norm_eps=0.0`，tilelang（`torch.ops.vllm.mhc_pre_tilelang`）与 aiter（`mhc_pre(..., norm_weight, norm_eps)`）都已实现；
  上游 `_apply_mhc_norm` 在 `norm_weight=None` 时**直接返回 layer_input**，即「不做 norm」本身也是合法调用方式（`vllm/models/deepseek_v4/amd/model.py:816` 就是按 `fuse_mhc_rmsnorm` 决定传 None 还是传权重）。
  因此本需求是把 `dlc_mhc_pre` 的签名补齐到与上游一致，而不是给旧算子塞一个只属于 GLM5 的分支：`norm_weight=None` 时逐位等价于现状，其他调用方零影响。
- **调用方范围（DLC 侧）**：mHC 只出现在 GLM5 系列（`vllm/models/glm5next/*`，45 层 × 2 site）；DeepSeek-V4 的 mHC 走 tilelang / AMD / XPU 后端，不在 DLC 路径上，`vllm-cl` 里也没有对应的适配。
  当前 DLC 上 `MHCPreOp` 走的是纯 PyTorch 的 `forward_native`，插件用 `forward_oot` 在外面补 RMSNorm（`vllm_cl/patch/patch_mhc.py:16`）；接入 `dlc_mhc_pre` 后这步应进 kernel。
- 复现脚本：`probes/probe_mhc_pre_post.py`。
- 本次调整（2026-09-22）：需求类型「新增算子」→「扩展已有算子能力」；优先级 P1 不变；算子名称明确为扩展 `torch.dlc_mhc_pre`（增加 `norm_weight` / `norm_eps`）。
- 二次复核（2026-09-22）：**维持「扩展」**，理由见上（上游所有后端都有该可选参数，`None` 时行为不变）；同批复核中 #16 `silu_and_mul_with_clamp` 已由「扩展」改为「新增独立算子」，两者判据不同（见 16 号文件）。

## 优先级
P1

## 目标版本与期望时间
GLM-5.3-Flash 适配 Phase 2（性能阶段）。正确性 bring-up 已在 native 上完成，本需求面向生产性能；具体里程碑待算子组排期确认。

## 算子名称
扩展 `torch.dlc_mhc_pre`（增加 `norm_weight` / `norm_eps`）

## 背景与功能说明
使用场景：GLM-5.3-Flash 全部 45 层都使用 4 路 mHC（hyper-connection）残差流，每层 attention 前与 FFN 前各有一个 pre site，即 90 次/step。
当前问题：DLC 上走 `MHCPreOp.forward_native` → `mhc_kernels.mhc_pre_torch`，是纯 PyTorch 组合实现：RMS 归一化、`[N,16384] × [16384,24]` 的映射 GEMM、pre/post sigmoid、`softmax + 20 次 Sinkhorn` 迭代、四路 collapse。**每次 20 次 Sinkhorn 迭代会产生大量小 kernel**，45 层 × 2 site 下 launch 开销和 HBM 往返非常显著。
另外插件已发现上游 `forward_native` **不处理 `norm_weight`/`norm_eps`**：GLM-5 会在 collapse 之后对 `layer_input` 做子层 RMSNorm，插件目前用 OOT wrapper 在返回前补上（`vllm_cl/patch/patch_mhc.py`）。新的 DLC 算子应把这步 RMSNorm 纳入 kernel，避免额外一次读写。
目标功能：单 kernel（或少量 kernel）完成 RMS → 24 维映射 → pre/post/comb 计算 → Sinkhorn(20) → collapse → 子层 RMSNorm。
非目标：post 侧残差合并（见 14 号）；post+pre 融合（见 15 号）。

## 期望 API 与函数签名
> **本条是「扩展已有算子」**：算子名保持 `torch.dlc_mhc_pre`，只是在末尾增加可选参数
> `norm_weight` / `norm_eps`（默认 `None` / `0.0`，为 `None` 时逐位等价于现状）。

```python
def mhc_pre(
    residual: torch.Tensor,     # [N, Hc, D] bf16, Hc=4, D=4096
    fn: torch.Tensor,           # [mix, Hc*D] fp32, mix=(2+Hc)*Hc=24, 即 [24, 16384]
    hc_scale: torch.Tensor,     # [3] fp32
    hc_base: torch.Tensor,      # [mix] fp32, 即 [24]
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,  # GLM-5.3-Flash=2.0
    sinkhorn_repeat: int,       # 20
    norm_weight: torch.Tensor | None = None,  # [D] bf16, 子层 RMSNorm 权重
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 返回 (post_mix [N,Hc,1] fp32, comb_mix [N,Hc,Hc] fp32, layer_input [N,D] bf16)
```

## 参数与返回值
输入：
- `residual`：`[N, 4, 4096]`，BF16，四路残差流。
- `fn`：`[24, 16384]`，FP32，映射权重（checkpoint 的 `hc_attn_fn` / `hc_ffn_fn`）。
- `hc_scale`：`[3]`，FP32（对应 pre / post / comb 三段）。
- `hc_base`：`[24]`，FP32。
- `rms_eps` / `hc_pre_eps` / `hc_sinkhorn_eps`：FP32，GLM-5.3-Flash 分别为 `1e-6`（hc_eps）、pre eps、sinkhorn eps。
- `hc_post_mult_value`：默认 2.0。
- `sinkhorn_repeat`：默认 20（`hc_sinkhorn_iters`）。
- `norm_weight` / `norm_eps`：可选，子层 RMSNorm；GLM-5 会传（这是上游 native 缺失的部分）。

返回值：
- `post_mix`：`[N, 4, 1]`，FP32。
- `comb_mix`：`[N, 4, 4]`，FP32。
- `layer_input`：`[N, 4096]`，BF16，**已应用子层 RMSNorm**（与上游 native 行为不同，需在本 kernel 内完成）。

边界：`N == 0` 返回空；`norm_weight=None` 时跳过 RMSNorm。

## 功能语义
```text
flat      = X.reshape(N, 16384).float()                     # 4 路展平
mix_logits= (flat @ fn.T) * rsqrt(mean(flat^2) + rms_eps)   # [N, 24]
pre       = sigmoid(mix_logits[:, 0:4]  * scale[0] + base[0:4]) + eps
post      = hc_post_mult_value * sigmoid(mix_logits[:,4:8] * scale[1] + base[4:8])
C         = (mix_logits[:,8:24].reshape(N,4,4) * scale[2] + base[8:24])
comb      = Sinkhorn(softmax(C, dim=-1) + eps, repeat=20)   # 行列交替归一化
x_in      = sum_i pre[:, i] * X[:, i]                       # [N, 4096]
layer_input = RMSNorm(x_in, norm_weight, norm_eps)           # 子层 RMSNorm（上游 native 缺失）
```

关键点：
1. `mix_logits` 与 Sinkhorn 全程 FP32；`pre/post/comb` 输出保留 FP32（下游还要用）。
2. Sinkhorn 20 次迭代必须在 kernel 内完成，不得展开成 20 组 elementwise op。
3. `norm_weight/norm_eps` 必须生效——这是与上游 `mhc_pre_torch` 的差异点。

broadcasting：`hc_base` 分段广播到 0:4 / 4:8 / 8:24。
in-place / alias：不修改输入。
空 / 零长度：`N == 0` 返回空。
异常：`residual.shape` 与 `fn.shape` 不匹配、`hc_base` 长度不为 24 报错。
确定性：无随机。

## 支持范围与约束
支持的 dtype：`residual`/`norm_weight` BF16；`fn/hc_scale/hc_base` FP32；输出 `post_mix/comb_mix` FP32、`layer_input` BF16。
支持的 shape：`N` 动态（1~32K）；`Hc=4`；`D=4096`（GLM-5.3-Flash 固定，建议不写死）；`mix=24`。
数值范围：`mix_logits` 由 FP32 GEMM 产生；Sinkhorn 迭代过程需数值稳定（`+eps`）。
layout：`residual` 连续 `[N,Hc,D]`；`fn` 连续 `[24, 16384]`。
device：DLC 单卡。
执行模式：decode 路径需支持 CUDA graph。
其他限制：`norm_weight` 语义与 `_apply_mhc_norm` 一致（FP32 归约、乘权重、cast 回 BF16）。

## 精度要求
Golden / reference：`mhc_kernels.mhc_pre_torch`（GLM 调用）+ 插件 OOT wrapper 的 RMSNorm（`vllm_cl/patch/patch_mhc.py:16` 的 `_rms_norm`）。
输入分布：`residual ~ N(0,1)`，`fn ~ N(0, 1/16384)`；另加 `mix_logits` 极端值样本。
样本数量：≥ 200 组。
比较 dtype：`layer_input` BF16；`post_mix/comb_mix` FP32。
atol / rtol：`layer_input atol=1e-2, rtol=1e-2`；`post_mix/comb_mix atol=1e-4, rtol=1e-4`。
必须逐元素一致：Sinkhorn 输出（`comb` 每行每列和为 1，误差需在 eps 量级）；`norm_weight=None` 时必须与不带 RMSNorm 的 reference 一致。
NaN / Inf / 极值：Sinkhorn 不发散；全 0 输入时输出有界。
累加精度：GEMM、softmax、Sinkhorn、collapse 全程 FP32。

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
1. 功能：N ∈ {1, 64, 4096}；`norm_weight` 提供 / 不提供。
2. `hc_post_mult_value` ∈ {1.0, 2.0}；`sinkhorn_repeat` ∈ {1, 20}。
3. 边界：N=0、全 0 residual、`fn` 中带极端值。
4. 精度：Sinkhorn 收敛性（行/列和误差）；FP32 中间量统计。
5. 异常 / layout / CUDA graph。
6. 性能：N=4096、90 个 site/step 的总耗时；目标相对 native（含 20 组 elementwise）≥ 5×。

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
Reference 实现：`vllm/model_executor/layers/mhc.py:96`（`MHCPreOp`）的 `forward_native` → `mhc_kernels.mhc_pre_torch`。
插件补丁（RMSNorm 缺失）：`vllm_cl/patch/patch_mhc.py:28`（`_pre_oot_with_norm`）。
调用点：GLM-5.3-Flash `vllm/models/glm5next/nvidia/model.py` 的 HC 层。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md` §7.2/§7.3（推荐 OOT 算子 `dlc_mhc_pre`）。
配置：`hc_mult=4`、`hc_sinkhorn_iters=20`、`hc_eps=1e-6`。
