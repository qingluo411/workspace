# [Operator Request] DLC iHC head 融合算子

## 需求类型

新增算子。

## 优先级

P1。每个 token 只执行一次，正确性可先由 eager fallback 保证；但它是最终 logits
之前的必经路径，生产性能验收前应完成融合。

## 目标版本与期望时间

Hy4-preview-Mix DLC 性能优化第一阶段；不阻塞首轮全模型跑通。

## 算子名称

- 建议 PyTorch 接口：`torch.dlc_ihc_head`。
- 建议 kernel 名称：`custom_dlc_ihc_head_bf16`。
- 本需求只包含最终 4 通道 residual 的 RMS + gate + 降维。

## 背景与功能说明

使用场景：

- 78 层 decoder 完成后，hidden states 仍为 `[T,4,6144]`。
- iHC head 用 RMSNorm 后的 4 路 gate 将通道合并为 `[T,6144]`，再进入 final norm
  和 lm_head。

当前问题：

- eager 路径包含 flatten、FP32 cast、RMS、FP32 GEMM、sigmoid、broadcast multiply、
  sum、BF16 cast，约 15 次 launch。
- 与 pre 相比执行次数少，但仍是每 token 固定开销，并产生大中间张量。

目标功能：

- 一次 launch 完成 RMS、4 路 gate 投影、sigmoid gate 和加权合并。

非目标：

- 不实现 pre/post。
- 不融合 final RMSNorm 或 lm_head。
- 不支持 `hc_mult` 非 4。

## 期望 API 与函数签名

```python
def dlc_ihc_head(
    x: torch.Tensor,             # [T, 4, 6144], bfloat16
    head_weight: torch.Tensor,   # [4, 24576], float32
    head_scale: torch.Tensor,    # [1], float32
    head_base: torch.Tensor,     # [4], float32
    hc_eps: float,
    rms_eps: float,
) -> torch.Tensor:
    ...
```

建议 C++ 签名：

```cpp
at::Tensor dlc_ihc_head(
    const at::Tensor& x,
    const at::Tensor& head_weight,
    const at::Tensor& head_scale,
    const at::Tensor& head_base,
    double hc_eps,
    double rms_eps);
```

返回 `[T,6144]` BF16。

## 参数与返回值

- `x`：BF16 `[T,4,6144]`。
- `head_weight`：FP32 `[4,24576]`。
- `head_scale`：FP32 `[1]`。
- `head_base`：FP32 `[4]`。
- `hc_eps`：Hy4 为 `1e-6`；`rms_eps`：Hy4 为 `1e-5`。
- 返回 BF16 `[T,6144]`。

## 功能语义

```text
xf    = float32(x.reshape(T, 4 * 6144))
rsqrt = rsqrt(mean(xf * xf, dim=-1) + rms_eps)
mixes = head_weight @ xf.T * rsqrt             # [T,4]
gate  = sigmoid(mixes * head_scale[0] + head_base) + hc_eps
out   = sum_i gate[:, i] * x[:, i, :]          # FP32 累加，最后 BF16
```

要求：

- RMS 均值维度为 `4*6144`。
- gate、加权和在 FP32 中完成，最后一次性舍入到 BF16。
- `T=0` 返回空张量。

## 支持范围与约束

- DLC/TYD；`T=0..4096`；`hc_mult=4`；`hidden_size=6144`。
- contiguous 输入/权重；单卡；graph capture。
- 不允许 host sync 或 D2H。

## 精度要求

- Golden：`HYV4HCHeadLayer.forward` 的 FP32 reference。
- BF16 输出：`atol=2e-2`，`rtol=2e-2`。
- 每组 shape 至少 100 组随机样本，覆盖接近零、极值、NaN/Inf 传播。

## 测试范围

- [x] 基本功能与 reference 对齐
- [x] 多 dtype 精度
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [ ] 非 contiguous、特殊 stride 或 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式
- [x] 长时间稳定性
- [x] 性能基准与性能回归

用例：`T = 0,1,2,7,16,32,128,1024,4096`；全 0 gate、极大 gate、典型 checkpoint
权重；graph replay。

## 验证环境与依赖版本

- vLLM `108b49ea10`；vllm-cl `89731b3`；PyTorch `d0221a52cc`。
- Driver `26.1.7`；CRT `0TYD`；RTABI `3.2.0`。
- DLC_Custom_Kernel `9bcbb426e`；LLVM `66c6fd00f204`。

## 验收标准

- [ ] API 和参数校验符合约定。
- [ ] `T=0..4096` 精度通过。
- [ ] 单次 launch 完成 head。
- [ ] graph replay 稳定。
- [ ] 文档、测试和示例已提交。

## Reference、日志与附件

- Reference：`vllm/vllm/models/hy_v4/nvidia/hc.py::HYV4HCHeadLayer.forward`
- 调用方：`HYV4Model.forward` 的 `self.hc_head(hidden_states)`。

