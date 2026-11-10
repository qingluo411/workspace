<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22）：可复用已有算子（本条撤回，不提交） -->
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：已被 dlc_mhc_post 完全覆盖（建议撤回，不要提单）
     证据：实测 out_err=3.90e-03（bf16 级），语义与 native 一致；comb_res_mix 需 flat [..., hc_mult*hc_mult]，x 为 [..., hidden_size]
     建议：撤回本需求；插件直接调用 dlc_mhc_post（把 dlc_mhc_pre 返回的 cm 原样传入） -->

# [Operator Request] dlc_mhc_post（mHC 后置残差合并）

## 需求类型
其他

**需求分类：可复用已有算子（本条撤回，不提交）**
- 现有算子：`torch.dlc_mhc_post(x, residual, post_layer_mix, comb_res_mix)`（实现：`aten/src/ATen/native/dlc/mhc_post.cxx`），语义 `out = post_layer_mix * x + comb_res_mix.T @ residual`。
- 实测（dlc:0）：`out_err=3.90e-03`（bf16 量级），与 native reference 一致 → 已完全覆盖。
- 接入注意（否则会报参数错误）：`comb_res_mix` 需 **flat** `[..., hc_mult*hc_mult]`（可直接传 `dlc_mhc_pre` 返回的 `cm`）；`x` 为 `[..., hidden_size]`（单路输入，不是 `[..., hc_mult, hidden_size]`）；`x`/`residual` 为 bf16，两个 mix 为 fp32。
- 复现脚本：`probes/probe_mhc_pre_post.py`。
- 本次调整（2026-09-22）：需求类型「新增算子」→「其他（复用已有算子）」；优先级 P1→P2；算子名称改为直接调用 `torch.dlc_mhc_post`；**本条撤回，不提交**。

## 优先级
P2

## 目标版本与期望时间
本条不提交给算子组、不需要排期：插件侧在 Phase 2 内把 hc_post 切到现有 `torch.dlc_mhc_post`（算子组无需改动）。
## 算子名称
复用 `torch.dlc_mhc_post`（原提名 `dlc_mhc_post` 撤回，插件直接调用即可）

## 背景与功能说明
使用场景：mHC 的 post site 把子层输出按 `post_mix` 写回四路残差流，并按 `comb_mix` 做流间混合，45 层 × 1 次/层。
当前问题：DLC 走 `MHCPostOp.forward_native` → `mhc_kernels.mhc_post_torch`，由若干 elementwise + 广播乘加组成，每次都读写 `[N,4,4096]` 的 BF16 残差流（N=4096 时约 128 MiB/次），是显著的 HBM 带宽开销。
目标功能：单 kernel 完成 `residual_out = post * x + comb^T @ residual`。
非目标：pre 侧计算（13 号）；与 pre 的融合（15 号）。

## 期望 API 与函数签名
> **本条已撤回（不提交）**：下面这段签名是初稿拟定的形式，仅作存档；实际动作是复用现有算子 `torch.dlc_mhc_post`
> （见「需求分类」段与「算子名称」），**不需要算子组提供这个新 API**。

```python
def mhc_post(
    x: torch.Tensor,            # [N, D] bf16, 子层输出
    residual: torch.Tensor,     # [N, Hc, D] bf16, 当前四路残差流
    post_layer_mix: torch.Tensor,  # [N, Hc, 1] fp32
    comb_res_mix: torch.Tensor,    # [N, Hc, Hc] fp32
) -> torch.Tensor:              # [N, Hc, D] bf16
```

## 参数与返回值
输入：
- `x`：`[N, 4096]`，BF16。
- `residual`：`[N, 4, 4096]`，BF16。
- `post_layer_mix`：`[N, 4, 1]`，FP32（来自 13 号或上一 site）。
- `comb_res_mix`：`[N, 4, 4]`，FP32。

返回值：
- `residual_out`：`[N, 4, 4096]`，BF16。

边界：`N == 0` 返回空；不修改输入（如需原位写残差流，请明确提供 out 参数）。

## 功能语义
```text
residual_out[n, j, :] = post_layer_mix[n, j] * x[n, :]
                      + sum_i comb_res_mix[n, i, j] * residual[n, i, :]
```

broadcasting：`post` 沿 D 维广播，`comb` 沿 D 维广播。
in-place / alias：不修改输入；输出新分配（若提供 out 缓冲则原位写）。
空 / 零长度：`N == 0` 返回空。
异常：形状不符报错。
确定性：无随机；累加 FP32。
累加精度：按 head 求和用 FP32，输出 cast 回 BF16。

## 支持范围与约束
支持的 dtype：`x`/`residual` BF16；`post_layer_mix`/`comb_res_mix` FP32；输出 BF16。
支持的 shape：`N` 动态（1~32K）；`Hc=4`；`D=4096`。
数值范围：常规。
layout：`residual` 连续 `[N,Hc,D]`。
device：DLC 单卡。
执行模式：decode 路径需支持 CUDA graph。
其他限制：输出精度对残差链敏感——四路残差会跨 45 层累积，输出 BF16 的舍入误差会被逐层放大，请评估是否需要用 FP32 中间量 + 单次 cast（当前 native 即如此）。

## 精度要求
Golden / reference：`mhc_kernels.mhc_post_torch`。
输入分布：`x ~ N(0,1)`、`residual ~ N(0,1)`、`comb` 为行归一化矩阵、`post ~ (0,2)`。
样本数量：≥ 200 组。
比较 dtype：BF16 输出对 FP32 reference。
atol / rtol：`atol=1e-2, rtol=1e-2`。
必须逐元素一致：`comb = I`、`post = 0` 时输出必须严格等于输入 residual（bit-exact）。
NaN / Inf / 极值：不产生 NaN。
累加精度：FP32 累加，单次 cast。

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
1. 功能：N ∈ {1, 64, 4096}；`comb` 为单位阵 / 随机行归一化矩阵。
2. 恒等性：`comb=I, post=0` 时输出与输入 residual bit-exact。
3. 长链验证：连续 45 层残差累积后的误差统计（评估 BF16 存储影响）。
4. 边界 / 异常 / CUDA graph。
5. 性能：N=4096 单次耗时与带宽占用；目标 ≥ 3×（对比 native 多算子）。

## 验证环境与依赖版本
DLC 硬件：8 卡，设备名 `/dev/cltech0-7`（TP=8，无 PP）
vLLM commit：`f5361fb09f`
vLLM DLC（vllm-cl）commit：`a14fc5b`
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：`VLLM_USE_DLC_COL_MAJOR_MATMUL=1`
模型：GLM-5.3-Flash-Mix（AWQ 4-bit，`linear_num_heads=64`、`linear_head_dim=128`、`short_conv_kernel_size=4`、`gate_lower_bound=-5.0`）

## 验收标准
- [ ] 插件侧用 `torch.dlc_mhc_post` 替换 native 实现，`comb_res_mix` 传 **flat** `[..., hc_mult*hc_mult]`（与 `dlc_mhc_pre` 返回的 `cm` 直接对接），`x` 为单路 `[..., hidden_size]`。
- [ ] 接入后逐元素一致（实测 `out_err=3.901e-03`，bf16 量级）。
- [ ] 45 层 × 2 site 的 launch 数与耗时记录，确认相比 native 组合的收益。
- [ ] 不修改现有算子：本条不产生算子组接口/行为变更。
## Reference、日志与附件
Reference 实现：`vllm/model_executor/layers/mhc.py:312`（`MHCPostOp.forward_native`）→ `mhc_kernels.mhc_post_torch`。
调用点：GLM-5.3-Flash `vllm/models/glm5next/nvidia/model.py`。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md` §7.2/§7.3。
