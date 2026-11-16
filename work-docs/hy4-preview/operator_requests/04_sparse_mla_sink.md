# [Operator Request] DLC sparse MLA learnable sink epilogue

## 需求类型

新增算子。

## 优先级

P0。Hy4 的 `learnable_sink_param` 是架构的一部分；当前 DLC sparse MLA 路径没有
sink 消费方，若静默退化为无 sink，会系统性改变每层 attention 输出。

## 目标版本与期望时间

Hy4-preview-Mix DLC bring-up 第一阶段。先提供独立 epilogue，复用现有 DLC sparse
attention kernel；后续再评估把 sink 融合进主 attention kernel。

## 算子名称

- 建议 PyTorch 接口：`torch.dlc_sparse_mla_apply_sink_`。
- 建议 kernel 名称：`custom_dlc_sparse_mla_apply_sink_bf16`。
- 本需求只包含 sink 对 attention 输出的缩放；不包含 top-k、attention 主计算、
  DCP combine 或 gated MLA。

## 背景与功能说明

使用场景：

- Hy4 每个 attention head 有一个 FP32 learnable sink，共 64 头；TP8 下每 rank
  8 个头。
- 上游 FlashMLA sparse 路径把 sink 折入 softmax 分母：
  `out *= exp(lse) / (exp(lse) + exp(sink))`。
- DLC `torch.dlc_sparse_attention_kv_cache` 已支持 head_dim 576、top-k 2048、
  BF16 KV cache 和可选 LSE，但没有 sink 参数。

当前问题：

- DLC sparse MLA backend 会忽略上游传入的 `sinks`。
- 没有 sink 时 attention 输出整体偏大，且偏差随 head/序列变化，不能靠后处理
  gate 修正。
- 如果把 sink 直接塞进主 sparse attention kernel，需要覆盖全部 head/top-k/
  max-token 变体；首期成本过高。

目标功能：

- 对已有 attention 输出和 LSE 做一次设备侧 epilogue：
  `out *= sigmoid(lse - sink)`。
- 支持在 DCP combine 之后统一应用，避免每个 rank 重复计算 sink。
- 返回未修改的 LSE，供后续 combine/调试使用。

非目标：

- 不修改 sparse attention 主 kernel。
- 不实现 DCP combine 本身。
- 不实现 gated MLA 或 o_proj。
- 不改变 top-k 选择。

## 期望 API 与函数签名

```python
def dlc_sparse_mla_apply_sink_(
    attn_out: torch.Tensor,    # [T, H, 512], bfloat16, in-place
    softmax_lse: torch.Tensor, # [T, H], float32 优先，支持 bfloat16
    attn_sink: torch.Tensor,   # [H], float32
) -> torch.Tensor:
    ...
```

建议 C++ 签名：

```cpp
at::Tensor& dlc_sparse_mla_apply_sink_(
    at::Tensor& attn_out,
    const at::Tensor& softmax_lse,
    const at::Tensor& attn_sink);
```

返回 `attn_out` 本身，标记为 mutation。

## 参数与返回值

- `attn_out`：BF16 `[T,H,512]`，DLC sparse attention 的 latent 输出；
  Hy4 TP8 时 `H=8`，TP1 调试时可为 64。
- `softmax_lse`：`[T,H]`，自然对数 LSE；优先 FP32，兼容 BF16。必须是应用 sink
  之前的原始 LSE。
- `attn_sink`：FP32 `[H]`，允许 `-inf`。
- 返回：原地修改后的 `attn_out`。

调用约定：适配层必须让 sparse attention 返回原始 LSE，并优先使用现有
`lse_dtype=torch.float32` 参数；若只能得到 BF16 LSE，需要重新评估本算子的
`atol/rtol`，不得默认使用 BF16 LSE 后仍按 FP32 精度验收。

## 功能语义

对每个 `(t,h)`：

```text
lse = float32(softmax_lse[t,h])
s   = float32(attn_sink[h])
factor = exp(lse) / (exp(lse) + exp(s))
       = sigmoid(lse - s)
attn_out[t,h,:] *= bf16(factor)
```

边界语义：

- `lse = -inf`（该 token/head 无有效 top-k）：`factor = 0`，输出置 0。
- `s = -inf` 且 `lse` 有限：`factor = 1`，输出不变。
- `lse = +inf` 或 NaN：按数值异常处理；不得产生未定义内存写。
- 所有计算用 FP32；最终缩放结果舍入到 BF16。
- `attn_out` 和 `softmax_lse` 的 head 维必须一致。
- `T=0` 直接返回，不启动 kernel。

## 支持范围与约束

- 设备：DLC/TYD。
- `attn_out`：BF16，`[T,H,512]`，contiguous；`T=0..4096`。
- `softmax_lse`：FP32/BF16，`[T,H]`，contiguous。
- `attn_sink`：FP32，`[H]`，contiguous。
- 首期 `H=8`（TP8）为发布路径；`H=64` 用于 TP1 单卡/调试。
- 只支持 `head_dim_v=512`、单卡 TP 内本地 head；不涉及 DCP 通信。
- 支持 graph capture；不得包含 host sync、`.item()`、动态分支依赖 tensor 值。
- 非 contiguous 输入首期不支持。

## 精度要求

- Golden：PyTorch FP32 参考，严格按 `sigmoid(lse - sink)` 计算后再舍入 BF16。
- 输入分布：`lse` 在 `[-50, 50]`、接近 0、`-inf`；sink 在
  `[-100, -10, -1, 0, 1, 10, 100, -inf]`。
- `attn_out`：normal、uniform、极小值、极大值、NaN/Inf 传播。
- 每个 shape 至少 100 组，固定 seed。
- 输出 BF16：`atol=2e-2`，`rtol=2e-2`；同时报告 factor 的 FP32 误差。
- 必须覆盖“所有 top-k 无效”的全 mask 行，输出必须为 0 而非 NaN。

## 测试范围

- [x] 基本功能与 reference 对齐
- [x] 多 dtype 精度（BF16 out + FP32/BF16 LSE + FP32 sink）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [ ] 非 contiguous、特殊 stride 或 layout（首期不支持）
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式
- [x] 长时间稳定性
- [x] 性能基准与性能回归

具体用例：

- `T = 0,1,2,7,16,32,128,1024,4096`；`H=8`、`H=64`；`D=512`。
- sink 取 `-inf`、大负数、0、大正数。
- LSE 取 `-inf`、`-1000`、`-50`、0、50、1000、NaN。
- 部分 head 全 mask、全部 head 全 mask。
- graph replay 100 次，原地 buffer 不被重复缩放。

## 验证环境与依赖版本

- DLC 硬件：TYD，8 卡，单卡 64 GiB。
- vLLM commit：`108b49ea10`；vLLM DLC：`89731b3`。
- PyTorch/torch-dlc：`2.5.0`，pytorch `d0221a52cc`。
- Driver：`26.1.7`；CRT `0TYD`；RTABI `3.2.0`。
- DLC_Custom_Kernel：`9bcbb426e`；LLVM：`66c6fd00f204`。

## 验收标准

- [ ] API、in-place mutation 和异常行为符合约定。
- [ ] `T=0..4096`、`H=8/64` 精度通过。
- [ ] 全 mask 行输出 0，不产生 NaN。
- [ ] graph replay 幂等性由调用方保证；单次调用不重复应用。
- [ ] 文档、测试和示例已提交。

## Reference、日志与附件

- Reference：`vllm/vllm/models/hy_v4/nvidia/flashmla_sparse.py`
  （`out *= exp(lse) / (exp(lse) + exp(sink))` 的语义）
- 现有 DLC attention：`vllm-cl/vllm_cl/attention/sparse_mla.py::_forward_bf16_kv`
- 现有 DLC op：`pytorch/aten/src/ATen/native/dlc/FusedOps.cpp::dlc_sparse_attention_kv_cache`
- 调用点：`vllm/vllm/models/hy_v4/nvidia/attention.py::HYV4MLAAttention`

