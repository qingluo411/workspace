# [Operator Request] DLC gated MLA epilogue

## 需求类型

新增算子。

## 优先级

P1。正确性可由 `linear_gate` + `sigmoid` + elementwise multiply 的 eager 路径
保证；但 Hy4 有 78 个 MLA 层，每 token 都执行一次大 gate GEMM，生产性能验收前
应融合。

## 目标版本与期望时间

Hy4-preview-Mix DLC 性能优化第一阶段；不阻塞首轮全模型正确性跑通。

## 算子名称

- 建议 PyTorch 接口：`torch.dlc_gated_mla_epilogue`。
- 建议 kernel 名称：`custom_dlc_gated_mla_epilogue_bf16`。
- 本需求融合 gate projection、sigmoid 和 attention 输出相乘。

## 背景与功能说明

使用场景：

- Hy4 `gated_mla=true`，`gating_type="elementwise"`。
- `linear_gate` 将 `[T,6144]` hidden states 投影到 `[T,16384]`，TP8 下本地宽度
  `2048`；再与 MLA 输出逐元素相乘。
- 该 gate 权重不在 AWQ 量化排除列表之外，但配置显式将 `linear_gate` 放入
  `modules_to_not_convert`，因此是 BF16。

当前问题：

- eager 路径需要一次 `[T,6144] × [6144,2048]` GEMM、一次 sigmoid、一次 multiply。
- 中间 gate tensor `[T,2048]` 需要额外写回和读取；decode 时 launch 开销显著。
- 上游 CUDA 有 HPC gated MLA 融合，DLC 没有对应实现。

目标功能：

- 一次 launch 完成 `attn_out * sigmoid(hidden_states @ gate_weight.T)`。
- 支持 TP8 本地宽度 2048；保留 TP1 调试宽度 16384 的可选支持。

非目标：

- 不支持 `headwise` gating。
- 不融合 `o_proj`、MLA 主体或 sparse attention。
- 不支持量化/FP8 gate weight。

## 期望 API 与函数签名

```python
def dlc_gated_mla_epilogue(
    hidden_states: torch.Tensor,  # [T, 6144], bfloat16
    gate_weight: torch.Tensor,    # [N, 6144], bfloat16, row-major
    attn_out: torch.Tensor,       # [T, N], bfloat16
) -> torch.Tensor:
    ...
```

建议 C++ 签名：

```cpp
at::Tensor dlc_gated_mla_epilogue(
    const at::Tensor& hidden_states,
    const at::Tensor& gate_weight,
    const at::Tensor& attn_out);
```

返回 `[T,N]` BF16。

## 参数与返回值

- `hidden_states`：BF16 `[T,6144]`，contiguous。
- `gate_weight`：BF16 `[N,6144]`，row-major；TP8 时 `N=2048`，TP1 调试可为
  `N=16384`。
- `attn_out`：BF16 `[T,N]`，contiguous，MLA 输出。
- 返回新的 `[T,N]` BF16；不修改输入。

## 功能语义

```text
gate   = float32(torch.matmul(hidden_states.float(), gate_weight.float().T))
gate   = sigmoid(gate)
out    = bf16(attn_out.float() * gate)
```

要求：

- 无 bias。
- 使用 elementwise gating，不做 headwise broadcast。
- GEMM 累加和 sigmoid 使用 FP32，最终舍入到 BF16。
- `T=0` 返回空张量。

## 支持范围与约束

- 设备：DLC/TYD。
- `T=0..4096`；`K=6144`；`N=2048`（TP8 发布路径）；可选 `N=16384`（TP1）。
- 三个输入 BF16、contiguous。
- 单卡；N 必须与 `attn_out.shape[-1]` 一致。
- 支持 graph capture；不允许 host sync。
- 首期不支持非 contiguous、bias、headwise、量化 gate。

## 精度要求

- Golden：FP32 `attn_out * sigmoid(hidden_states @ gate_weight.T)`，最后 BF16。
- 输入分布：normal、uniform、接近零、极值；固定 seed。
- 每个 shape 至少 100 组；`atol=2e-2`，`rtol=2e-2`。
- 报告 GEMM 输出和最终输出 max abs/rel error；NaN/Inf 按 IEEE 传播。

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

具体用例：

- `T = 0,1,2,7,16,31,32,64,128,1024,4096`。
- `N=2048,16384`；gate 全 0、接近 0、极大值。
- graph replay 100 次。
- 与 eager 路径对比 launch 数和 decode latency。

## 验证环境与依赖版本

- vLLM `108b49ea10`；vllm-cl `89731b3`；PyTorch `d0221a52cc`。
- Driver `26.1.7`；CRT `0TYD`；RTABI `3.2.0`。
- DLC_Custom_Kernel `9bcbb426e`；LLVM `66c6fd00f204`。

## 验收标准

- [ ] API 和参数校验符合约定。
- [ ] TP8 `N=2048` 精度通过。
- [ ] 不产生额外 `[T,N]` gate 中间张量。
- [ ] graph replay 稳定。
- [ ] 文档、测试和示例已提交。

## Reference、日志与附件

- Reference：`vllm/vllm/models/hy_v4/nvidia/attention.py` 中非 HPC gated MLA 分支
- 上游融合语义：`vllm/vllm/model_executor/layers/hpc/gated_mla.py`
- 权重样例：`model.layers.0.self_attn.linear_gate.weight`

