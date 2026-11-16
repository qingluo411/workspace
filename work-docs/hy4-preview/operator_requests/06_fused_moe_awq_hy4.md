# [Operator Request] W4A16 AWQ MoE 支持 Hy4 FP32 router 与 clamp 10

## 需求类型

扩展已有算子能力并修复接入精度。

## 优先级

P0。Hy4 的 77 个 sparse 层全部依赖该 MoE 路径；当前 vllm-cl 接入会把
`swiglu_limit` 固定为 `7.0`，并把 FP32 router logits 转成 BF16，均与 checkpoint
配置不一致。

## 目标版本与期望时间

Hy4-preview-Mix DLC bring-up 第一阶段。优先修 vllm-cl 调用；若 256-expert 或
group64/TP8 形状暴露 kernel 问题，再补 kernel 回归。

## 算子名称

`torch.dlc_vllm_fused_moe_w4a16_awq_col_major`

现有公开接口不需要新增参数。本需求要求现有参数在 Hy4 配置下行为正确：

- `gating_output` 接受 FP32；
- `swiglu_limit=10.0`；
- `global_num_experts=256`、`topk=8`、group64 非对称 AWQ；
- `scoring_func="sigmoid"`、`renormalize=True`、`routed_scaling_factor=2.827`；
- `e_score_correction_bias` 为 FP32。

## 背景与功能说明

使用场景：

- Hy4 layer0 为 BF16 dense MLP；layer1..77 为 AWQ W4A16 sparse MoE。
- 每层 256 routed experts、top-8、expert intermediate 2048、group size 64、
  带 zero point。
- routed expert 使用 clamped SwiGLU：`gate=clamp(gate,max=10)`，
  `up=clamp(up,-10,10)`，`silu(gate)*up`；dense/shared expert 不 clamp。

当前问题：

1. `vllm-cl/vllm_cl/ops/awq_marlin.py` 的
   `DLCAWQMarlinMoEMethod.apply_monolithic` 把 `swiglu_limit=7.0` 写死。Hy4 的
   checkpoint 配置是 `10.0`，会改变 routed expert 激活。
2. 同一函数把 `gating_output` 强制 `.to(torch.bfloat16)`。Hy4 的
   `GateLinear` 输出和 `e_score_correction_bias` 为 FP32；提前舍入可能改变
   top-k 选点，尤其是专家分数接近时。
3. 现有 GLM 路径主要覆盖不同 expert 数/中间维；Hy4 需要新增 256-expert、
   group64、TP8（intermediate per partition 256）和 TP8+EP8（32 local experts，
   intermediate 2048）回归。

已确认的 kernel 能力：

- `pytorch/aten/src/ATen/native/dlc/FusedOps.cpp` 已接受
  `gating_output` 为 BF16 或 FP32。
- `DLC_Custom_Kernel/dlc_kernels/moe/router.hpp` 已支持
  `dlc_bf16`/`dlc_fp32` gating 和 FP32 `e_correction_bias`。
- kernel 已接收 `swiglu_limit` scalar 并传给 SwiGLU 预处理。

因此本需求预期不需要新 MoE kernel；重点是接入修复、能力确认和回归覆盖。

非目标：

- 不新增 W4A16/W8A16 量化格式。
- 不改变 AWQ 权重布局/预处理语义。
- 不实现 expert parallel 通信。
- 不实现 shared expert 融合；shared expert 仍走独立 BF16 MLP。
- 不优化 router gate GEMM；由 `DLCGateLinear` 负责。

## 期望 API 与函数签名

保持现有 Python/C++ 签名：

```python
def dlc_vllm_fused_moe_w4a16_awq_col_major(
    hidden_states: torch.Tensor,       # [T, 6144] bf16
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_qzeros: torch.Tensor,
    w2_qzeros: torch.Tensor,
    gating_output: torch.Tensor,       # [T, 256] fp32
    topk: int,                         # 8
    global_num_experts: int,           # 256
    expert_map: torch.Tensor | None,
    renormalize: bool,                 # True
    use_group_topk: bool,              # False for n_group=1
    topk_group: int | None,
    num_expert_group: int | None,
    scoring_func: str,                 # "sigmoid"
    routed_scaling_factor: float,      # 2.827
    e_score_correction_bias: torch.Tensor,
    apply_router_weight_on_input: bool,
    input_ids: torch.Tensor | None = None,
    hash_table: torch.Tensor | None = None,
    activation: str = "silu",
    swiglu_limit: float | None = 10.0,
) -> torch.Tensor:
    ...
```

调用方必须从 `layer.swiglu_limit`（或 `layer.moe_config.swiglu_limit`）读取 clamp，
不得再使用常量 `7.0`；`gating_output` 不得强制转 BF16。

## 参数与返回值

- `hidden_states`：BF16 `[T,6144]`，contiguous。
- `gating_output`：FP32 `[T,256]`，Hy4 发布路径；BF16 仍兼容。
- `e_score_correction_bias`：FP32 `[256]`。
- `w13_qweight` / `w2_qweight`：按当前 DLC AWQ col-major 布局预处理后的 I32。
- `w13_scales` / `w2_scales`：BF16。
- `w13_qzeros` / `w2_qzeros`：I32，非对称 AWQ 必填。
- 返回 BF16 `[T,6144]`。

## 功能语义

路由：

```text
logits = float32(gating_output)
scores = sigmoid(logits)
scores = scores + float32(e_score_correction_bias)
selected = topk(scores, 8)
weights = scores[selected]
if renormalize: weights = weights / sum(weights)
weights = weights * 2.827
```

专家：

```text
for each selected expert:
  gate_up = awq_dequant_gemm(hidden, expert.w13)
  gate, up = split(gate_up)
  gate = clamp(gate, max=10.0)
  up   = clamp(up, min=-10.0, max=10.0)
  act  = silu(gate) * up
  y   += expert_down(act) * routing_weight
```

shared expert 不加 clamp，也不应误走本算子的 routed clamp。

行为要求：

- top-k 和 sigmoid 在 FP32 中完成；不得先转 BF16。
- `swiglu_limit` 必须使用运行时值；`None` 表示不 clamp。
- `n_group=1, topk_group=1` 时按普通 top-k 处理。
- `T=0` 返回空张量，不启动 kernel。

## 支持范围与约束

发布形状：

- hidden size：6144。
- routed expert intermediate：2048。
- global experts：256；top-k：8。
- group size：64；4-bit；asymmetric zero point。
- TP8 no-EP：每 rank `num_experts=256`、`intermediate_per_partition=256`。
- TP8 + EP8：每 rank `num_experts=32`、`intermediate_per_partition=2048`。
- `T=0..4096`；graph capture 支持。
- `activation="silu"`，clamp 10.0。

其他：

- `expert_map` 支持 int32/int64；EP 时由 vLLM expert map 提供。
- 非 contiguous 输入首期不支持。
- 不要求支持 group32/group128 以外的 Hy4 专属变体；现有格式不回归。

## 精度要求

- Golden：FP32 router + 反量化 AWQ 权重 + 标准 SwiGLU clamp 的 PyTorch
  reference；专家计算可分批，但累加用 FP32。
- 路由：
  - 非并列情况下 top-k expert id 必须完全一致。
  - top-k weights：FP32，`atol=1e-4`，`rtol=1e-4`。
  - 并列分数单独记录，不在本需求中强行规定 tie-break 顺序。
- 最终 BF16 输出：`atol=2e-2`，`rtol=2e-2`。
- 输入分布：normal、uniform、接近零、极值、所有专家同分、单专家主导。
- 每个 T/shape 至少 100 组，固定 seed；报告 max abs/rel error。
- 必须对比 `swiglu_limit=10.0`、`7.0`、`None`，确认 10 与 7 的差异来自 clamp
  而非路由或布局错误。

## 测试范围

- [x] 基本功能与 reference 对齐
- [x] 多 dtype 精度
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [ ] 非 contiguous、特殊 stride 或 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [x] 多卡/分布式（TP8/EP8）
- [x] 长时间稳定性
- [x] 性能基准与性能回归

具体用例：

- `T = 0,1,2,7,16,31,32,64,128,256,1024,4096`。
- global experts 256；TP8 no-EP / TP8+EP8。
- group64，asym zero point；`swiglu_limit=10`、`=7`、`=None`。
- `gating_output` FP32 与 BF16 对比；FP32 必须是发布路径。
- `e_score_correction_bias` 全 0、随机、偏置明显。
- 单专家被选 8 次、8 个不同专家、全专家负载。
- graph replay 100 次。

## 验证环境与依赖版本

- DLC 硬件：TYD，8 卡，单卡 64 GiB。
- vLLM commit：`108b49ea10`；vLLM DLC：`89731b3`。
- PyTorch/torch-dlc：`2.5.0`，pytorch `d0221a52cc`。
- Driver：`26.1.7`；CRT `0TYD`；RTABI `3.2.0`。
- DLC_Custom_Kernel：`9bcbb426e`；LLVM：`66c6fd00f204`。
- 关键环境变量：`VLLM_PLUGINS=dlc,dlc_patches`、`DLC_TPU_VERSION=2`。

## 验收标准

- [ ] vllm-cl 不再硬编码 `swiglu_limit=7.0`。
- [ ] vllm-cl 不再把 FP32 router logits 强制转 BF16。
- [ ] 256 experts、group64、TP8/EP8 精度通过。
- [ ] `swiglu_limit=10` 的回归用例通过。
- [ ] 现有 GLM MoE 精度/性能无回归。
- [ ] 文档、测试和示例已提交。

## Reference、日志与附件

- 调用方：`vllm-cl/vllm_cl/ops/awq_marlin.py::DLCAWQMarlinMoEMethod.apply_monolithic`
- 问题行：同文件当前 `swiglu_limit=7.0` 和 `gating_output.to(torch.bfloat16)`。
- Kernel 入口：`pytorch/aten/src/ATen/native/dlc/FusedOps.cpp::dlc_vllm_fused_moe_w4a16_awq_col_major_impl`
- Router dtype 支持：`DLC_Custom_Kernel/dlc_kernels/moe/router.hpp`
- 模型调用方：`vllm/vllm/models/hy_v4/nvidia/moe.py::HYV4MoEFused`
- 权重样例：`model.layers.1.mlp.experts.0.{gate_proj,up_proj,down_proj}.*`

