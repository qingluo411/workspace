# [Operator Request] DLC iHC pre 融合算子

## 需求类型

新增算子。

## 优先级

P0。Hy4 decode 每个 token 在 78 层中执行 156 次 iHC pre；eager 路径会放大到
数千次 kernel launch，阻塞生产性能验收。

## 目标版本与期望时间

Hy4-preview-Mix DLC bring-up 第一阶段。目标是在 TP8 全模型跑通前具备单算子
替换能力；具体交付日期由算子组和接入方共同排期。

## 算子名称

- 建议 PyTorch 接口：`torch.dlc_ihc_pre`。
- 建议 kernel 名称：`custom_dlc_ihc_pre_bf16`。
- 本需求只包含 iHC pre：RMS 归一化、8 路 gate 投影、pre/post sigmoid gate、
  4 通道 pre 加权降维。
- 不包含 iHC post、head、跨层 post+pre 融合和后续 RMSNorm 折叠。

## 背景与功能说明

使用场景：

- Hy4 将标准单 residual stream 替换为 `hc_mult=4` 个并行 residual 通道。
- 每个 decoder 子块（attention、MLP）前后各有一个 iHC 边界。
- 单层 2 个边界，78 层共 156 次 pre；decode 每 token 均执行。

当前问题：

- 上游 eager 实现为 flatten、FP32 cast、RMS、FP32 GEMM、sigmoid、broadcast
  multiply、sum 等多步组合，单次 pre 约 20 个 kernel launch。
- DLC 没有 CUDA Triton/HPC 路径，只能走 eager fallback。
- 权重和激活规模较小，主要成本是 launch 和中间 tensor 往返，不是 FLOPs。

目标功能：

- 一次 device launch 完成 iHC pre 的完整语义。
- 输出 FP32 post gate，供后续 `dlc_ihc_post` 使用。
- 支持 decode/prefill 的动态 token 数，支持 graph capture。

非目标：

- 不实现 iHC post/head。
- 不把后续 RMSNorm 折叠进本算子。
- 不实现跨层 post+pre 融合。
- 不改变 Hy4 的 gate 公式或 epsilon 语义。

## 期望 API 与函数签名

```python
def dlc_ihc_pre(
    x: torch.Tensor,          # [T, 4, 6144], bfloat16
    fn_weight: torch.Tensor,  # [8, 24576], float32
    hc_scale: torch.Tensor,   # [2], float32
    hc_base: torch.Tensor,    # [8], float32
    magnitude: float,
    hc_eps: float,
    rms_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

建议 C++ 签名：

```cpp
std::tuple<at::Tensor, at::Tensor> dlc_ihc_pre(
    const at::Tensor& x,
    const at::Tensor& fn_weight,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    double magnitude,
    double hc_eps,
    double rms_eps);
```

返回：

- `reduced`: `[T, 6144]` BF16。
- `post`: `[T, 4]` FP32。

## 参数与返回值

输入：

- `x`：BF16，逻辑 shape `[T, 4, 6144]`，4 个 iHC 通道。
- `fn_weight`：FP32，shape `[8, 24576]`，前 4 行为 pre gate，后 4 行为 post gate。
- `hc_scale`：FP32，shape `[2]`，分别用于 pre/post gate。
- `hc_base`：FP32，shape `[8]`，前 4 项为 pre bias，后 4 项为 post bias。
- `magnitude`：Python float，Hy4 为 `2.0`。
- `hc_eps`：Python float，Hy4 为 `1e-6`。
- `rms_eps`：Python float，Hy4 为 `1e-5`。

返回值：

- `reduced`：BF16 `[T, 6144]`。
- `post`：FP32 `[T, 4]`。

## 功能语义

对每个 token 独立计算：

```text
xf     = float32(x.reshape(T, 4 * 6144))
rsqrt  = rsqrt(mean(xf * xf, dim=-1) + rms_eps)
mixes  = fn_weight @ xf.T * rsqrt             # [T, 8]

pre_raw  = mixes[:, 0:4]
post_raw = mixes[:, 4:8]

pre  = sigmoid(pre_raw  * hc_scale[0] + hc_base[0:4]) + hc_eps
post = magnitude * sigmoid(post_raw * hc_scale[1] + hc_base[4:8]) + hc_eps

reduced = sum_i pre[:, i] * x[:, i, :]        # FP32 累加，最后舍入到 BF16
```

行为要求：

- RMS 的均值维度是 `4 * 6144`，不是逐 channel 或逐 token 子块。
- `hc_eps` 在 sigmoid 之后相加；`post` 的 `magnitude` 只乘 sigmoid，不与
  `hc_eps` 相乘。
- gate 计算和加权累加使用 FP32；`reduced` 最后舍入到 BF16。
- `T=0` 时返回对应空张量，不启动 kernel。
- 输入不被修改，无 alias。

## 支持范围与约束

- 设备：DLC/TYD。
- `x`：BF16，shape `[T, 4, 6144]`，P0 支持 `T=0..4096`。
- `fn_weight`：FP32，shape 必须为 `[8, 24576]`。
- `hc_scale`：FP32 `[2]`；`hc_base`：FP32 `[8]`。
- `x` 的最后两维连续；P0 可要求整个 tensor contiguous。
- `fn_weight`、`hc_scale`、`hc_base` 必须 contiguous。
- 首期只支持 `hc_mult=4`、`hidden_size=6144`。
- 单卡；本算子不涉及 TP/EP，iHC 参数在每 rank 复制。
- 支持 `FULL_DECODE_ONLY` graph capture；不得包含 D2H copy、`.item()`、
  tensor truth-value 或 host synchronize。

## 精度要求

- Golden/reference：`vllm/vllm/models/hy_v4/nvidia/hc.py` 中 eager pre，
  全程 FP32 计算，最终 BF16 输出。
- Golden 运行设备：CPU/PyTorch F32 参考即可。
- 输入分布：`randn(0, 1)`、`randn(0, 0.02)`、均匀分布、接近零、含极值。
- 随机种子固定；每个 shape 至少 100 组样本。
- `reduced`：BF16，`atol=2e-2`，`rtol=2e-2`。
- `post`：FP32，`atol=2e-3`，`rtol=2e-3`。
- 必须报告 max abs/rel error；NaN/Inf 输入按 IEEE 传播，不允许静默改成 0。
- 中间 RMS、gate、加权和不得提前舍入到 BF16。

## 测试范围

- [x] 基本功能与 reference 对齐
- [x] 多 dtype 精度（输入 BF16，权重/门控 FP32，输出 BF16 + FP32）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [ ] 非 contiguous、特殊 stride 或 layout（P0 可要求 contiguous）
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式
- [x] 长时间稳定性
- [x] 性能基准与性能回归

具体测试用例：

- `T = 0, 1, 2, 7, 16, 31, 32, 64, 127, 128, 256, 1024, 4096`。
- `T` 非 8/16/32 对齐的尾块。
- graph capture 下同一 buffer 连续 replay 100 次，结果一致。
- 与 eager 路径做 kernel count / latency 对比，记录单次 launch 数。

## 验证环境与依赖版本

- DLC 硬件：TYD，8 卡，单卡 64 GiB。
- vLLM commit：`108b49ea10`（`upstream-v0.30.1rc0` 基线）。
- vLLM DLC commit：`89731b3`（Hy4 适配分支基线）。
- PyTorch/torch-dlc：`2.5.0`，pytorch main `d0221a52cc`。
- Runtime/driver：`Driver Version_v26.1.7`；CRT `0TYD`，RTABI `3.2.0`。
- Custom Kernel/LLVM：DLC_Custom_Kernel main `9bcbb426e`；LLVM main `66c6fd00f204`。
- 关键环境变量：`VLLM_PLUGINS=dlc,dlc_patches`、`DLC_TPU_VERSION=2`。

## 验收标准

- [ ] API、参数校验和错误行为符合约定。
- [ ] `T=0..4096` 基本功能和精度通过。
- [ ] post gate 的 FP32 精度达到约定。
- [ ] graph replay 稳定，无 host sync。
- [ ] 单次 launch 完成 pre，不产生不可接受的中间 HBM 往返。
- [ ] 文档、测试和示例已提交。

## Reference、日志与附件

- Reference：`vllm/vllm/models/hy_v4/nvidia/hc.py::HYV4HCPreLayer.forward`
- 现有相近 DLC 实现：`DLC_Custom_Kernel/dlc_kernels/vllm/mhc_pre.cpp`
  （DeepSeek V4 的 3×hc gate + sinkhorn 版本，语义不同，不能直接复用输出布局）
- 调用方：`vllm/vllm/models/hy_v4/nvidia/model.py::HYV4DecoderLayer._forward_ihc`
- 权重样例：`model.layers.0.hc_attn_layer.hc_pre.{hc_fn,hc_scale,hc_base}`

