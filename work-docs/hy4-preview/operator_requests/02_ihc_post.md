# [Operator Request] DLC iHC post 融合算子

## 需求类型

新增算子。

## 优先级

P0。每个 decoder 子块输出都要经过 iHC post；78 层每 token 共 156 次。eager 路径
需要 FP32 cast、broadcast multiply、add、cast 回 BF16，且会产生 `[T,4,6144]`
中间张量。

## 目标版本与期望时间

Hy4-preview-Mix DLC bring-up 第一阶段，与 `dlc_ihc_pre` 同期或紧随交付。

## 算子名称

- 建议 PyTorch 接口：`torch.dlc_ihc_post`。
- 建议 kernel 名称：`custom_dlc_ihc_post_bf16`。
- 本需求只包含 post gate 散射和多通道 residual 加法。

## 背景与功能说明

使用场景：

- attention 或 MLP 子块在单通道 `[T, 6144]` 上计算完成。
- iHC post 将子块输出写回 4 个 residual 通道：
  `y[n,i,d] = post[n,i] * x[n,d] + residual[n,i,d]`。

当前问题：

- eager 会先转 FP32，构造 `[T,4,6144]` 的 `post_gated` 中间张量，再加 residual，
  最后转 BF16。
- 该张量在 prefill 时可达数百 MiB，带来额外分配和读写。
- 每次 post 多次 launch，decode 路径累计开销大。

目标功能：

- 一次 launch 完成 4 通道门控散射和 residual 累加。
- 支持按需直接写入预分配输出，降低峰值内存。

非目标：

- 不计算 post gate；post gate 由 `dlc_ihc_pre` 输出。
- 不实现 attention/MLP 本身。
- 不实现跨层 post+pre 融合。

## 期望 API 与函数签名

```python
def dlc_ihc_post(
    x: torch.Tensor,         # [T, 6144], bfloat16
    residual: torch.Tensor,  # [T, 4, 6144], bfloat16
    post: torch.Tensor,      # [T, 4], float32
) -> torch.Tensor:
    ...
```

建议 C++ 签名：

```cpp
at::Tensor dlc_ihc_post(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post);
```

返回 `[T, 4, 6144]` BF16。

## 参数与返回值

输入：

- `x`：子块输出，BF16 `[T, 6144]`。
- `residual`：iHC 多通道 residual，BF16 `[T, 4, 6144]`。
- `post`：`dlc_ihc_pre` 输出的 FP32 gate，`[T, 4]`。

输出：

- `out`：BF16 `[T, 4, 6144]`。

首期 out-of-place；接口可预留 `out=` 版本，但不得改变返回值语义。

## 功能语义

```text
for n in 0..T-1:
  for i in 0..3:
    for d in 0..6143:
      out[n,i,d] = bf16(float32(post[n,i]) * float32(x[n,d]) + float32(residual[n,i,d]))
```

要求：

- 乘法与加法在 FP32 中完成，最后舍入一次到 BF16。
- 输入不修改；无 alias。
- `T=0` 返回空张量，不启动 kernel。
- 不允许出现 `post` 与 `x` 广播错位；`post` 的通道维必须映射到 residual 第 1 维。

## 支持范围与约束

- 设备：DLC/TYD。
- `T=0..4096`；`hc_mult=4`；`hidden_size=6144`。
- `x` 连续 `[T,6144]`；`residual` 连续 `[T,4,6144]`；`post` 连续 `[T,4]` FP32。
- 单卡；不涉及 TP/EP。
- 支持 graph capture；不得 host sync。
- 首期不要求非 contiguous stride。

## 精度要求

- Golden：`x.float() * post.float().unsqueeze(-1) + residual.float()`，再转 BF16。
- 输入分布：normal、uniform、接近零、极值；固定 seed；每个 shape 至少 100 组。
- BF16 输出：`atol=2e-2`，`rtol=2e-2`。
- 必须报告 max abs/rel error；NaN/Inf 按 IEEE 传播。

## 测试范围

- [x] 基本功能与 reference 对齐
- [x] 多 dtype 精度（BF16 激活 + FP32 gate）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [ ] 非 contiguous、特殊 stride 或 layout（P0 可要求 contiguous）
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式
- [x] 长时间稳定性
- [x] 性能基准与性能回归

具体用例：

- `T = 0, 1, 2, 7, 16, 31, 32, 64, 128, 256, 1024, 4096`。
- gate 全 0、全 1、接近 0、`magnitude` 后约 2 的典型值。
- residual 全 0、极大值、极小值、NaN/Inf 传播。
- graph replay 100 次一致。

## 验证环境与依赖版本

- DLC 硬件：TYD，8 卡，单卡 64 GiB。
- vLLM commit：`108b49ea10`；vllm-cl：`89731b3`。
- PyTorch：`2.5.0`，pytorch `d0221a52cc`。
- Driver：`26.1.7`；CRT `0TYD`；RTABI `3.2.0`。
- DLC_Custom_Kernel：main `9bcbb426e`；LLVM：main `66c6fd00f204`。

## 验收标准

- [ ] API 和参数校验符合约定。
- [ ] `T=0..4096` 精度通过。
- [ ] 不在 kernel 内产生额外 HBM 临时张量。
- [ ] graph replay 稳定。
- [ ] 文档、测试和示例已提交。

## Reference、日志与附件

- Reference：`vllm/vllm/models/hy_v4/nvidia/hc.py::HYV4HCPostLayer.forward`
- 现有相近 DLC 实现：`DLC_Custom_Kernel/dlc_kernels/vllm/mhc_post.cpp`
  （DeepSeek V4 含 comb mix，Hy4 无 comb，不能直接复用语义）
- 调用方：`HYV4HCLayer.post`

