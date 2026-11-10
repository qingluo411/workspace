# Kimi-K3 SiTU-GLU / AttentionRes 算子验收报告

日期：2026-11-09

## 1. 结论

| 算子 | 功能正确性 | 边界/副作用 | DLC Graph | 性能目标 | 结论 |
|---|---|---|---|---|---|
| `situ_and_mul` | 通过 | 通过 | 通过 | 未通过 | 功能可合入，性能未达验收线 |
| `kimi_k3_attn_res` | 未通过 | 通过 | 通过 | 未通过 | 数值精度与性能均未达验收线 |

`situ_and_mul` 的 BF16 数值、动态 shape、3D/空 tensor、非连续外层
stride、输入不变性和 caller `out` 均通过。`T=4096, 2D=12288` 的 eager
执行中位数为 3.37 ms，未达到需求文档的 1 ms，也未达到相对 native
组合 3x 的目标。

`kimi_k3_attn_res` 的 prefix 更新、residual bank 写回、空 tensor、非法
参数、固定标量 Graph capture/replay 均通过；但 `num_blocks` 大于 1 的
mixing 路径在部分输入和 rows 上超过 `atol=rtol=2e-2`。失败来自 QK
logit 使用 BF16 normalized K 计算，而需求要求 FP32 点积。

## 2. 验证基线

| 组件 | 分支 / 提交 |
|---|---|
| DLC_Custom_Kernel | `release-kimi-k3-int4` / `d8d11e14c` |
| pytorch | `release-kimi-k3-int4` / `a10e6dc403` |
| vllm | `release-kimi-k3-int4` / `d9230f277e` |
| vllm-cl | `release-kimi-k3-int4` / `60231f8` |
| 硬件 | 8 x TYD，真机，单卡 DLC device 0 |

构建日志：

- `/root/workspace/log/kimi-k3-operators-kernel-build-20261109-032147.log`
- `/root/workspace/log/kimi-k3-operators-pytorch-rebuild2-20261109-033420.log`

## 3. 集成修复

### 3.1 AttentionRes PyTorch 注册

源分支同时声明了 DLC 和 `CompositeImplicitAutograd`，但没有提供 CPU
composite 实现，导致 `libtorch_cpu.so` 链接时引用只存在于
`libtorch_dlc.so` 的符号，PyTorch 构建失败。

修复：

- `CompositeImplicitAutograd` 映射到新增的 CPU fallback
  `kimi_k3_attn_res_composite`；
- CPU fallback 位于 `aten/src/ATen/native/KimiK3AttnRes.cpp`；
- DLC dispatch 仍走 `dlc_kimi_k3_attn_res` 和
  `custom_attention_residual`。

### 3.2 SiTU 测试 API

文档期望 `linear_beta=None` 表示关闭 up soft clipping。自动生成的顶层
`torch.dlc_situ_and_mul` PythonArgParser 会把显式 `None` 变成默认的
25.0；`torch.dlc.situ_and_mul` 包装器通过直接调用 dispatcher，能正确
保留 None。

本次测试使用已实现的公开包装器 `torch.dlc.situ_and_mul`，并修正了
测试用例：

- 将错误调用 `torch.dlc_situ_and_mul` 改为 `torch.dlc.situ_and_mul`；
- 将非法 beta 从静态数值 case 移到独立 `pytest.raises` 测试，避免测试
  框架把预期异常当作失败。

顶层生成绑定的 optional-scalar 行为仍需后续统一，或在需求文档中明确
只支持 `torch.dlc.situ_and_mul`。

## 4. SiTU-GLU 结果

### 4.1 仓库测试

| 测试 | 结果 |
|---|---|
| `TestSituAndMulBf16 --static --perf` | 23/23 通过 |
| `TestSituAndMulBf16 --dynamic --iterations 10 --batch-size 5` | 50/50 通过 |
| 非法 beta / 奇数尾维 / FP32 输入 | 1 个 pytest case 通过 |

日志：

- `/root/workspace/log/kimi-k3-situ-static-rerun-20261109-034533.log`
- `/root/workspace/log/kimi-k3-situ-dynamic-rerun-20261109-034547.log`

### 4.2 文档补充验收

额外覆盖并通过：

- `T`：0、1、8、64、256、4096；
- `2D`：256、512、1024、2048、4096、12288；
- 3D 与 4D outer shape；
- `(beta, linear_beta)`：`(4,25)`、`(4,None)`、`(0.5,3)`、
  `(16,0.25)`、`(1,1)`、`(0.1,None)`；
- 全 0、空 tensor、NaN/Inf 传播；
- 非连续外层 stride 和 input storage offset；
- 输入不被修改，返回 tensor 与 caller `out` 地址一致。

### 4.3 Graph

固定输入和 `out` buffer 的 `torch.dlc.graph` capture/replay 通过，
replay 后与 FP32 reference 最大绝对误差 `4.88e-4`，在 `1e-2` 容差内。

### 4.4 性能

测试 shape：`T=4096, 2D=12288, D=6144`，warmup 3 次，事件计时 10 次。

| 实现 | 中位数 |
|---|---:|
| DLC `situ_and_mul` | 3.37 ms |
| PyTorch native 组合 | 4.95 ms |

相对 native 约 `1.47x`，未达到 3x；单次 3.37 ms，也未达到 1 ms。

## 5. AttentionRes 结果

### 5.1 仓库测试

| 测试 | 结果 |
|---|---|
| `TestAttentionResidual --static` | 1/2 通过；T=128,n=8,outnorm=on 失败 |
| `TestAttentionResidual --dynamic --iterations 4 --batch-size 4` | 16/16 通过，但本次随机样本均落在 T=1 |

静态失败 case 的 prefix 与 block 写回均为 bit-exact，仅输出 tensor 的
3254/917504 个元素超过容差，最大误差 `1.10e-1`。日志：

- `/root/workspace/log/kimi-k3-attnres-static-20261109-034605.log`
- `/root/workspace/log/kimi-k3-attnres-dynamic-20261109-034616.log`

### 5.2 文档矩阵补充验收

固定随机种子，测试 `T ∈ {1,8,64,256,4096}`，覆盖：

- `num_blocks=0/1/7/8`；
- `block_write_idx=-1/0/7`；
- `delta` 有/无；
- `output_norm_weight` 有/无；
- H=7168，B=8。

结果：

- 16 个配置中 12 个通过，4 个失败；
- `num_blocks=0` 和 `num_blocks=1` 在所有测试 T 上通过且 bit-exact；
- `num_blocks=7` 在 T=64/256 失败；
- `num_blocks=8` 在 T=256/4096 失败，T=64 本次通过；
- 所有 16 个配置的 prefix 更新和 residual bank 写回均 bit-exact；
- `output_norm_weight=None` 时同样复现数值失败，排除 output RMSNorm
  是唯一根因。

典型失败：

| T | n | outnorm | 超容差元素 | 最大误差 |
|---:|---:|---|---:|---:|
| 64 | 7 | on | 7331 / 458752 | 0.2528 |
| 256 | 7 | on | 21975 / 1835008 | 0.7658 |
| 256 | 8 | on | 13589 / 1835008 | 0.2416 |
| 4096 | 8 | on | 391723 / 29360128 | 1.2720 |

### 5.3 精度根因

Kernel 将 `v * rstd * norm_weight` 先量化为 BF16，再用 BF16 QK
计算 logits。需求 reference 则要求
`sum(values_fp32 * norm_weight_fp32 * qk_weight_fp32) * rstd` 使用 FP32
点积。两者在部分 rows 上会使 softmax 概率分布明显偏移。

复现样本（T=64, n=1, row=14）：

- FP32 reference logits：`[8.6711, 9.6123]`；
- BF16-normalized QK 模拟 logits：`[8.6779, 9.8837]`；
- softmax 从 `[0.2807, 0.7193]` 偏移到 `[0.2304, 0.7696]`；
- kernel 输出与该 BF16 模拟最大差 `0.0156`；
- kernel 输出与 FP32 reference 最大差 `0.3059`，超过 `2e-2`。

因此失败不是简单的 launch race 或副作用错误，而是当前 QK 精度路径与
需求不符。仅提高输出 tolerance 不是正确修复。

### 5.4 CUDA 参考实现的 QK 数据类型

NVIDIA SM100 的生产 kernel `csrc/libtorch_stable/kimi_k3/attn_res_kernel.cu`
并不使用 BF16 乘法：

- `rms_w`、`res_w` 从 BF16 转成 FP32 后相乘，得到 FP32 `q_cache`；
- residual value 从 BF16 转成 FP32；
- QK 点积使用 `float2_fma`，即 FP32 乘加和 FP32 累加；
- RMS 统计、`rstd`、logit、softmax 和 PV 累加均为 FP32。

AMD/Triton 路径同样是先 `.to(tl.float32)`，再用 FP32 计算
`norm_weight * qk_weight` 和 `sum(values * weight)`。因此 CUDA 和 Triton
的共同语义是：**存储/搬运是 BF16，QK 乘法与累加是 FP32**。

### 5.5 符合 DLC 限制的修正方案

如果 DLC 矩阵单元只能接受 BF16 操作数，不能直接做 FP32 QK，可以使用
double-BF16 分解模拟 FP32 score weight：

```text
q_fp32 = float(norm_weight) * float(qk_weight)
q_hi   = bf16(q_fp32)
q_lo   = bf16(q_fp32 - float(q_hi))
dot    = bf16_qk(raw_v_bf16, q_hi) + bf16_qk(raw_v_bf16, q_lo)
logit  = dot * rstd_fp32
```

两个 BF16 QK 都使用 FP32 accumulator；softmax 和 PV 保持 FP32。这样不再
把 `rstd`、`norm_weight` 和原始 value 先合并后量化为 BF16，也只用 BF16
乘法满足硬件约束。

CPU 模拟结果（与验收同分布）：

| 方案 | 代表配置 | 超容差元素 | logit 最大误差 |
|---|---|---:|---:|
| CUDA 等价 FP32 q/dot | T=64,n=1; T=64,n=7/8; T=256,n=8 | 0 | 0 |
| current BF16-normalized K | 同上 | 8832 / 6163 / 11715 / 19536 | 最高约 0.54 |
| single BF16 q | 同上 | 5341 / 7700 / 13457 / 28995 | 最高约 0.61 |
| BF16 q hi+lo, FP32 accumulate | 同上 | 0 | <= 6.1e-5 |

结论：验收确实要求 CUDA 等价的 FP32 点积语义，但这不等价于必须在 DLC
矩阵单元直接执行 FP32 乘法。推荐采用 BF16 hi+lo 两次 QK 的方案，而不是
继续使用 BF16 normalized K。

### 5.6 Graph

T=8、B=8、n=8、固定 delta/block/out 和 scalars 的 capture/replay 通过：

- replay 输出超容差元素 0/57344；
- 最大误差 `1.54e-2`，在 `2e-2` 容差内；
- prefix 更新与 block 写回均 bit-exact。

### 5.7 性能

固定 n=8、delta/outnorm 均开启，事件计时 warmup 3 次：

| T | 中位数 | 需求 | 结果 |
|---:|---:|---:|---|
| 1 | 0.113 ms | <= 0.050 ms | 未通过 |
| 4096 | 40.78 ms | <= 10 ms | 未通过 |

## 6. 最终验收判断

- SiTU-GLU：BF16 功能和边界验收通过，Graph 通过；性能未通过。
- AttentionRes：API/副作用/Graph 通过；`num_blocks>1` 的数值精度未通过，
  性能未通过。
- 两个算子均不应在本轮按“需求文档全量验收通过”关闭；SiTU 可以按
  “正确性通过、性能待优化”单独推进，AttentionRes 需要先修正 QK 精度。
