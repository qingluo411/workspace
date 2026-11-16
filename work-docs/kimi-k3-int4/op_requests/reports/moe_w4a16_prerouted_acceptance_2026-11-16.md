# fused_moe_w4a16_prerouted_col_major 算子验收报告

日期：2026-11-16

## 1. 结论

算子主体功能和性能达到首期目标：

- FP32/INT32 路由、SiTU、compressed-tensors INT4 rowpack/col-major 布局均工作；
- `expert_map=None`、完整 global->local map、稀疏 map 和 `-1` 跳过均通过；
- 重复 expert route、`topk_ids=-1`、全 null route、`situ_linear_beta=None` 均通过；
- Graph capture/replay 通过；
- `T=1` 和 `T=64` 的生产形状性能明显优于目标。

但有一项需求未完全满足：

- wrapper 每次调用仍会 clone route 表并分配多块 device scratch，不是持久
  workspace；包括 `selected_experts`、`route_weights`、`route_offsets`、
  `expert_tokens`、`gathered_output`。这不会触发 host D2H，性能当前也达标，
  但不符合“内部 grouped-token scratch 必须持久/预分配、模型编排边界不产生额外
  copy”的验收条款。

另外，T=4096 未完成逐元素 reference 验证；真实 Kimi-K3 checkpoint 单 expert
bit-exact 验证因本机当前没有 `/data/models/Kimi-K3-INT4*` 无法执行。

## 2. 分支与构建

| 仓库 | 分支 | 结果 |
|---|---|---|
| DLC_Custom_Kernel | `release-kimi-k3-int4` | 从最新 `main` 重建，应用 `jinmnag-261112-feat-moe_w4a16_prerouted` |
| pytorch | `release-kimi-k3-int4` | 从最新 `main` 重建，应用同名远端分支 |

旧 Kimi 分支均备份为：

- `backup/release-kimi-k3-int4-20261116`

因前一问题 uv 已将本地 torch 替换成官方 2.10，本次恢复为 DLC torch：

```text
torch 2.5.0
torch.dlc.is_available() == True
```

构建日志：

- `/root/workspace/log/kimi-k3-moe-prerouted-kernel-build-20261116-031420.log`
- `/root/workspace/log/kimi-k3-moe-prerouted-pytorch-build-20261116-031534.log`
- `/root/workspace/log/kimi-k3-moe-prerouted-pytorch-develop-20261116-032122.log`

同事测试文件中的 variant/Python API 名称与实际注册不一致，已修正：

- kernel variant：`custom_vllm_fused_moe_w4a16_sym_col_major_g128_prerouted`
- Python API：`torch.dlc_vllm_fused_moe_w4a16_prerouted_col_major`

## 3. 仓库测试

| 测试 | 结果 |
|---|---|
| `TestFusedMoeW4A16Prerouted --static` | 7/7 通过 |
| `TestFusedMoeW4A16Prerouted --dynamic -i 5 -b 4` | 20/20 通过 |

日志：

- `/root/workspace/log/kimi-k3-moe-prerouted-static-20261116-032151.log`
- `/root/workspace/log/kimi-k3-moe-prerouted-dynamic-20261116-032213.log`

## 4. K3 生产形态补充验收

固定：

```text
H=3584, I_tp=384, K=16
global experts=896
local expert 数缩小为 8，以降低 reference 成本
topk_weights FP32, topk_ids INT32
```

完整 CPU reference 对比：

| T | expert_map | 超容差元素 | 最大误差 |
|---:|---|---:|---:|
| 1 | 896->8 map，含 -1 | 0 / 3584 | 0 |
| 64 | 896->8 map，含 -1 | 0 / 229376 | 0.00098 |
| 256 | 896->8 map，含 -1 | 0 / 917504 | 0.00195 |
| 64 | None | 0 / 229376 | 0.00195 |
| 64 | None，`situ_linear_beta=None` | 0 / 229376 | 0.00195 |

测试同时包含：

- 每个 batch 的重复 expert route；
- `topk_ids=-1`；
- `expert_map=-1`；
- caller `out` 原位写入、返回同一地址；
- 调用前后 `topk_ids` 不被修改；
- 全 `-1` route 输出严格为 0；
- 输入/输出无 NaN/Inf。

异常路径：

- `topk_weights` BF16、`topk_ids` INT64、错误 activation、错误 topk shape 均在
  launch 前报错；
- 越界正 expert id 在 kernel 中触发 `invalid pre-routed expert id` assertion，
  后续 API 返回 `synErrorLaunchFailure`；
- `T=0` 返回 `[0,3584]`。

## 5. Graph

固定输入、权重、route 表、`expert_map=None` 和 caller `out` 的 capture/replay 通过：

- replay 与 reference 超容差元素 0；
- 最大误差 `1.53e-5`；
- capture 内 wrapper 的 device scratch 分配被 graph allocator 接受；
- replay 后输出地址保持不变。

## 6. 性能

生产形状 `H=3584, I_tp=384, E=896, K=16, expert_map=None`，事件计时：

| T | 中位数 | 需求 | 结果 |
|---:|---:|---:|---|
| 1 | 0.467 ms | ≤4 ms/层 | 通过 |
| 64 | 11.50 ms | 24 层 ≤1 s（即 ≤41.7 ms/层） | 通过 |

按 24 层外推：T=1 约 11.2 ms，T=64 约 276 ms。

## 7. 剩余工作

1. 将 wrapper 中的 route scratch 改为由模型/runner 持有的持久 workspace，
   至少消除可避免的 `route_weights.clone()`；`topk_ids` 的 writable copy 也应复用。
2. 接入 vLLM FusedMoE/Native fallback 替换路径，验证真实 K3 layer 输出和整链
   graph。
3. 在存在真实 checkpoint 的环境中补单 expert bit-exact 和完整 W4A16 权重验证。
4. T=4096 的逐元素 reference 验证未完成；当前完整 reference 在该 PyTorch CPU
   构建上退化到单核，成本不可接受。
