<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22）：完全新算子 -->
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：未覆盖（保留新增，P2 且需评估收益）
     证据：DLC 只有独立 dlc_mhc_pre / dlc_mhc_post，无 fused post+pre；hasattr(torch,'dlc_mhc_fused_post_pre') 为 False
     建议：保留新增；建议先落地 pre/post 的 norm 扩展，再评估残差流往返收益是否值得融合 -->

# [Operator Request] dlc_mhc_fused_post_pre（mHC post+pre 融合，减少残差流往返）

## 需求类型
新增算子

**需求分类：完全新算子**
- 核对结果（2026-09-21）：DLC 只有独立的 `dlc_mhc_pre` / `dlc_mhc_post`，没有 fused post+pre 版本（`hasattr(torch, 'dlc_mhc_fused_post_pre') == False`），kernel 清单中亦无对应实现。
- 因此本需求保持「新增算子」（P2）。但建议先按 13/14 号接入独立 pre/post 并实测残差流往返占比：若融合收益 < 15%，可直接放弃本需求。
- 本次调整（2026-09-22）：需求类型与优先级（P2）均不变；建议先按 13/14 接入独立 pre/post 并实测残差流往返占比，收益 < 15% 则放弃本条。

## 优先级
P2

## 目标版本与期望时间
GLM-5.3-Flash 适配 Phase 2（性能阶段）。正确性 bring-up 已在 native 上完成，本需求面向生产性能；具体里程碑待算子组排期确认。

## 算子名称
`torch.dlc_mhc_fused_post_pre`（上游 `MHCFusedPostPreOp`）

## 背景与功能说明
使用场景：mHC 的 post site 与下一个 pre site 之间没有其他依赖，可以融合成一次 kernel，避免四路残差流（`[N,4,4096]` BF16）在两个 site 之间来回读写。GLM-5.3-Flash 45 层约 44 个相邻 site 可融合。
当前问题：DLC 走 `MHCFusedPostPreOp.forward_native`，即 post 与 pre 的 PyTorch 组合，残差流多次进出 HBM；N=4096 时单个 site 的残差流读写约 128 MiB，融合后可省掉一次完整往返。
目标功能：单 kernel 完成 `post（写回残差流） + pre（读残差流、RMS、映射、Sinkhorn、collapse、子层 RMSNorm）`。
非目标：单独优化 post 或 pre（13/14 号已覆盖）。
优先级说明：P2。正确性阶段用 13/14 两个算子即可；只有在 13/14 落地后仍因残差流往返受限时再投入。

## 期望 API 与函数签名
```python
def mhc_fused_post_pre(
    x: torch.Tensor,                # [N, D] bf16, 上一子层输出
    residual: torch.Tensor,         # [N, Hc, D] bf16
    post_layer_mix: torch.Tensor,   # [N, Hc, 1] fp32
    comb_res_mix: torch.Tensor,     # [N, Hc, Hc] fp32
    fn: torch.Tensor,               # [24, 16384] fp32
    hc_scale: torch.Tensor,         # [3] fp32
    hc_base: torch.Tensor,          # [24] fp32
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 返回 (residual_cur [N,Hc,D] bf16, post_mix [N,Hc,1] fp32,
    #       comb_mix [N,Hc,Hc] fp32, layer_input [N,D] bf16)
```

## 参数与返回值
参数语义与 13/14 号完全一致，额外要求：
- `x`、`residual` 为输入，`residual_cur` 为 post 后的残差流（可原位写 `residual` 以省内存，请明确是否允许）。
- `layer_input` 必须包含 `norm_weight/norm_eps` 的子层 RMSNorm（与 13 号一致，上游 native 缺失该步）。
- `n_splits` / `tile_n` 等调度参数不在本接口暴露，由算子内部决定。

边界：`N == 0` 返回空；`norm_weight=None` 跳过 RMSNorm。

## 功能语义
等价于依次执行 14 号（`mhc_post`）与 13 号（`mhc_pre`）：

```text
residual_cur = post(x, residual, post_layer_mix, comb_res_mix)
(post_mix, comb_mix, layer_input) = pre(residual_cur, fn, ...)
             # layer_input 含子层 RMSNorm
```

融合后中间量 `residual_cur` 可以停留在片上，不必完整写回 HBM（若必须写回，请说明融合带来的实际收益）。
空 / 零长度：`N == 0` 返回空。
异常 / 确定性 / 精度：同 13/14 号。

## 支持范围与约束
支持的 dtype / shape / layout / device / 执行模式：同 13/14 号。
其他限制：
1. 禁止使用 tilelang/aiter 等 NVIDIA 专用后端，必须提供 DLC 实现。
2. 若 `residual_cur` 需要写回，请明确写回的是原位 buffer 还是新分配，避免与调用方的 buffer 生命周期冲突。
3. 与 13/14 号必须数值等价（同一输入下三者输出一致）。

## 精度要求
Golden / reference：`vllm/model_executor/layers/mhc.py:450`（`MHCFusedPostPreOp.forward_native`）以及 13/14 号的 reference。
输入分布：同 13/14 号。
样本数量：≥ 200 组。
比较 dtype：同 13/14 号。
atol / rtol：同 13/14 号。
必须逐元素一致：**与 13 号 + 14 号串行执行的结果 bit-exact**（这是融合算子的核心正确性判据）。
NaN / Inf / 极值：同 13/14 号。

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
1. 等价性：同一输入下，`fused_post_pre` 与 `post → pre` 串行结果完全一致（含 `residual_cur`、`post_mix`、`comb_mix`、`layer_input`）。
2. 功能 / 边界 / 异常 / CUDA graph：同 13/14 号。
3. 性能：N=4096 下与「14 号 + 13 号」串行对比，给出残差流 HBM 流量下降比例与端到端收益；若收益 < 15% 请在评审中说明是否值得保留。

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
Reference 实现：`vllm/model_executor/layers/mhc.py:450`（`MHCFusedPostPreOp`），其 `forward_native` 为 post+pre 组合。
调用点：GLM-5.3-Flash `vllm/models/glm5next/nvidia/model.py`。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md` §7.3 推荐 OOT 算子 `dlc_mhc_fused_post_pre`。
