<!-- 按算子 Issue Form 字段编写，可直接粘贴到 GitHub Issue -->
<!-- 生成日期：2026-11-19 -->
<!-- 需求分类：需修复已有算子问题（数值确定性/精度） -->
<!-- 优先级：P0 -->

# [Operator Request] dlc_vllm_fused_moe_w4a16_awq_col_major 在不同 token 数下输出不确定

## 需求类型

**修复已有算子问题（数值确定性与精度）**

- 不新增算子，不改变 Python API。
- 修复对象：`torch.dlc_vllm_fused_moe_w4a16_awq_col_major`。
- 同一行输入、同一组 top-k expert/权重，仅因总 token 数从 `T=1` 变为 `T=2`，
  该行 routed expert 输出就可能出现 1 个 BF16 ULP 差异。

## 优先级

**P0**。该问题破坏输出确定性，并会把误差传给后续算子。

## 目标版本与期望时间

`DLC_Custom_Kernel` `release-glm53flash-3` 后续版本；具体排期请算子组确认。

## 算子名称

- Python operator：`torch.dlc_vllm_fused_moe_w4a16_awq_col_major`
- 主要 kernel：`custom_vllm_fused_moe_w4a16_col_major_g128` 及
  `align/decode/actorder_dynamic` 变体。
- 最低修复范围为 AWQ W4A16 `g128` 路径；同实现的 `g32/g64`、W8A16 和
  index-shuffle 路径建议一起审计。

## 问题描述

实际输入配置：

- `hidden_size=4096`
- `num_experts=288`
- `topk=8`
- 同一行输入的 top-k expert ids/weights 在 `T=1`、`T=2` 下完全相同。

第二个 token 激活额外 expert 后，当前调用的 active expert 总数变化：

| 场景 | active experts | 动态 `split_expert` | 目标行的 8 个 expert 分核 |
|---|---:|---:|---|
| `T=1` | 8 | 80 | XYS0 `4` / XYS1 `4` |
| `T=2` | 13 | 97 | XYS0 `5` / XYS1 `3` |

目标行 top-k expert：

```text
[47, 70, 73, 74, 80, 97, 117, 141]
```

第二个 token top-k expert：

```text
[2, 70, 73, 97, 106, 159, 219, 280]
```

因此目标行被分到不同的 XYS 组合：

```text
T=1: XYS0 [47,70,73,74] + XYS1 [80,97,117,141]
T=2: XYS0 [2,47,70,73,74,80] + XYS1 [97,106,117,141,159,219,280]
```

实测 gate logits 和 shared expert 输出一致，但目标行 routed expert 输出出现差异：

```text
T=1 = 0.027099609375
T=2 = 0.0269775390625
```

差异为 1 个 BF16 ULP。

## 当前实现与根因

`batch_fused_experts` 当前按 active expert 数量动态划分两个 XYS：

```text
active_experts = count(expert_size[i] > 0)
split_expert   = 使累计 active expert 数首次超过 active_experts / 2 的 expert id
XYS0 = [0, split_expert)
XYS1 = [split_expert, num_experts)
```

每个 XYS 将自己负责的 expert 输出累加到独立的 FP32 CMEM 区域，最后由 device 0
合并两个 CMEM 并转换成 BF16。

因此，active expert 总数变化会移动分核边界，使同一个 token 的 expert 输出采用不同的
FP32 accumulation grouping/order。本例中 `active_experts` 从 8 变为 13，分核点从 80
移到 97，目标行从 `4/4` 变为 `5/3`，最终有元素跨过 BF16 舍入边界。

问题不在 router、top-k 或 shared expert，而在 routed expert 的累加顺序。

## 建议修复方式

### 方案 A：固定 expert 到 XYS 的映射（建议先落地）

让分核只依赖 expert id，不依赖当前 active expert 集合：

```cpp
const int split_expert = num_experts / 2;
int num_experts_start = (device_id == 0) ? 0 : split_expert;
int num_experts_end   = (device_id == 0) ? split_expert : num_experts;
```

这样仍保留 `expert_size[i] == 0` 的稀疏跳零，但同一 token 始终进入相同 XYS，并保持
固定累加顺序。

代价：失去原来的 active-expert 动态均分。旧逻辑也只均衡 active expert 数量，并未按
实际 token 数/FLOPs 加权。建议进一步采用固定奇偶映射、固定交织/置换表或按真实激活
频率标定的静态双分桶，但映射必须与本次 token 集合无关。

### 方案 B：保留动态分核，固定最终 reduction 顺序

如果需要保留 dynamic split，则 expert 输出先写 per-expert staging，再按固定 expert
id 或固定 top-k rank 顺序归并，不能直接相加两个 XYS 的 partial sum。该方案会增加
暂存、带宽和归并成本，请算子组评估。

不建议直接恢复原动态分核而不处理最终 reduction 顺序，这正是本 issue 的根因。

## 修复要求

- 相同输入、相同 batch shape：bit-exact deterministic。
- 同一行输入在不同 `T` 下：该行 routed expert 输出的累加顺序必须一致。
- 不改变 router、top-k、shared expert、activation、routing scale 的数学语义。
- 不增加 host 同步、`.cpu()`、`.item()` 或逐 token Python 循环。
- 支持 eager 和 graph capture/replay。

## 算子级验收

- [ ] 构造 `x2 = cat([x0, x1])`，保证 `x0` 在 `T=1`、`T=2` 下 top-k ids/weights
  相同，但 `x1` 引入额外 active expert；比较两次输出第 0 行，要求 bit-exact。
- [ ] 随机覆盖 `active_experts=8/9/13/16`，至少重复 1000 次，第 0 行输出全部
  bit-exact。
- [ ] 覆盖 `topk=8`、`num_experts=288`、`hidden_size=4096` 的 W4A16 `g128` 路径。
- [ ] Graph capture/replay 下结果与 eager 一致。
- [ ] 记录 8/64/128-token kernel 耗时，并给出固定映射下的双 XYS 负载分布。

## 本地临时验证

本地将分核改为固定 `expert_id < num_experts/2` 后，同一行输入在 `T=1` / `T=2`
下不再出现该 routed expert ULP 差异。该改动仅用于验证根因和方向，最终实现及性能
评估请算子组完成。
