<!-- GitHub Issue Form：算子需求字段 -->
<!-- 生成日期：2026-10-28 -->
<!-- 需求分类：新增算子；优先级：P1 -->

# [Operator Request] dlc_situ_and_mul（Kimi SiTU-GLU）

## 需求类型

新增算子

## 优先级

P1

理由：Kimi-K3 的 dense MLP、2 个 shared experts 和所有 routed experts 都使用 SiTU-GLU。
当前 native 实现是 FP32 强转、tanh、sigmoid、可选 linear_beta tanh、乘法和 cast 的
多 kernel 组合。即使 #01 MoE 在 expert 内融合 SiTU，dense/shared 路径仍需要独立算子。

## 目标版本与期望时间

vLLM DLC `v0.30.1rc0` / Kimi-K3 性能里程碑；日期待算子组排期确认。

## 算子名称

`torch.dlc_situ_and_mul`

可参考上游 CUDA/XPU `torch.ops._C.situ_and_mul(out, x, beta, linear_beta)` 的语义。

## 背景与功能说明

使用场景：

- Kimi-K3 dense MLP：每层一次；
- Kimi-K3 shared experts：每层一次（2 个 shared expert 合并后的 MLP）；
- routed experts：可内联在 #01 MoE，也可用本算子作为 BF16 参考/小 batch fallback；
- K3 参数：`beta=4.0`、`linear_beta=25.0`。

当前问题：

- DLC 已有 `silu_and_mul` 和 `silu_and_mul_with_clamp`，语义分别是 SiLU 和 clamp-SwiGLU，
  不能计算 `beta*tanh(gate/beta)*sigmoid(gate)`；
- native `SituAndMul.forward_oot` 产生 gate/up FP32 临时张量并串联多次 elementwise；
- 在 prefill T=4096 时，shared expert 输出宽度较大，纯 elementwise 路径的 HBM 往返会放大。

目标功能：

- 单 kernel 计算 SiTU-GLU，支持可选 linear_beta；
- 输入为 gate/up 沿最后一维 packed 的 tensor；
- 不修改输入，输出同 dtype。

非目标：

- 不做 GEMM；
- 不做 clamp SwiGLU；
- 不包含 expert routing；
- 不要求反向训练算子。

## 期望 API 与函数签名

```python
def dlc_situ_and_mul(
    x: torch.Tensor,                 # [..., 2D], BF16/FP16/FP32
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
    *,
    out: torch.Tensor | None = None, # [..., D]，同 dtype
) -> torch.Tensor:
    ...
```

## 参数与返回值

输入 / 属性：

- `x`：最后一维 `2D`，前一半为 gate，后一半为 up；
- `beta`：gate 的 soft clipping 参数；K3 为 4.0；
- `linear_beta`：up 的 soft clipping 参数；K3 为 25.0，`None` 表示 up 不缩放；
- `out`：可选输出缓冲。

返回值：

- `[..., D]`，shape 为输入的最后一维减半，dtype 与输入一致。

行为：

- 不修改 `x`；
- 输入最后一维必须为偶数；输出宽度 `D` 推荐为 128 的倍数；
- `D=0` 返回空。

## 功能语义

```text
D = x.shape[-1] // 2
gate = x[..., :D].float()
up   = x[..., D:].float()

gate_act = beta * tanh(gate / beta) * sigmoid(gate)
if linear_beta is not None:
    up_act = linear_beta * tanh(up / linear_beta)
else:
    up_act = up

out = (gate_act * up_act).to(x.dtype)
```

数值要求：

- gate/up 的 tanh/sigmoid 和乘法使用 FP32；
- 最后只 cast 一次；
- `beta=0` 非法；
- `linear_beta<=0` 可作为“未设置”的 ABI 约定，但 Python wrapper 应把 `None` 统一映射为
  `-1.0` 或明确的 optional 参数；
- 无随机，确定性输出。

## 支持范围与约束

首期精确范围：

- dtype：BF16；建议同时支持 FP16；
- `beta=4.0`，`linear_beta=25.0`；
- `D` 为 128 的倍数；
- `x` 可为 2D `[T,2D]` 或 3D `[B,S,2D]`，最后一维连续；
- 总元素数动态，建议覆盖 T=1、8、64、256、4096；
- 单卡 DLC；
- 支持 eager，目标 graph capture。
- 调用方应优先提供 `out`，避免每层重新分配输出；本算子不得改变输入。

可选扩展：

- FP32 输入；
- 任意 shape，只要最后一维偶数；
- `out` 是非连续 view 时的支持不要求。

## 精度要求

Golden / reference：

- 以本 issue 末尾的自包含 PyTorch reference 为准。

输入分布：

- gate/up：`N(0,1)`、`U(-10,10)`、含大幅值和 0；
- beta/linear_beta 使用 K3 的 4/25，以及边界 1/1、beta=0.1、linear_beta=None。

比较：

- 输出转 FP32：`atol=1e-2, rtol=1e-2`；
- 全 0 输入输出必须为 0；
- NaN/Inf 按 PyTorch 语义传播，不允许静默 clamp 掉；
- 输入不被修改。

## 测试范围

- [x] 基本功能与 reference 对齐
- [x] 多 dtype 精度
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非 contiguous、特殊 stride 或 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式（逐元素算子）
- [x] 长时间稳定性
- [x] 性能基准与性能回归

具体测试用例、shape、数据范围和预期结果：

1. `[T,2D]`：`T ∈ {0,1,8,64,4096}`，`2D ∈ {256,512,1024,2048,4096,12288}`。
2. `[B,S,2D]`：至少一组 3D case。
3. 覆盖 `linear_beta=None`、`beta=4,linear_beta=25`、异常 `beta=0`。
4. 输入首地址偏移 / 非连续 batch stride。
5. 性能：T=4096、2D=12288 时目标 ≤ 1 ms；比 native 组合至少快 3×。

## 验证环境与依赖版本

```text
DLC 硬件：8 x TYD；driver v26.0.1 / Api 26
vLLM：85fd014d5ff4
vLLM DLC（vllm-cl）：ab819da9a0f4 + K3 适配工作区改动
PyTorch / torch-dlc：2.5.0，9049914ec94a
DLC_Custom_Kernel：55b9aef7e0cb
模型：Kimi-K3-INT4；hidden_act=situ，activation_situ_beta=4，activation_situ_linear_beta=25
```

## 验收标准

- [ ] API、参数校验和错误行为符合约定。
- [ ] BF16/FP16 精度满足 `atol=1e-2, rtol=1e-2`。
- [ ] `linear_beta=None` 时 up 原样通过。
- [ ] 输出等价于 native reference，输入不被修改。
- [ ] T=4096、2D=12288 单次 ≤ 1 ms（首期目标）。
- [ ] 可被 DLC Graph capture。
- [ ] 提供 `out` 时输出直接写入静态 buffer，不产生额外的 contiguous/copy 后处理。
- [ ] 文档、测试和示例已提交。

## Reference、日志与附件

### Self-contained PyTorch reference

```python
import torch


def situ_and_mul_reference(
    x: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError("last dimension must be even")
    split = x.shape[-1] // 2
    gate = x[..., :split].float()
    up = x[..., split:].float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(x.dtype)
```

模型调用形态：

```python
gate_up = gate_up_proj(hidden_states)[0]  # [T, 2D]
activated = situ_and_mul_reference(
    gate_up,
    beta=4.0,
    linear_beta=25.0,
)                                         # [T, D]
output = down_proj(activated)[0]
```
