<!-- GitHub Issue Form：算子需求字段 -->
<!-- 生成日期：2026-10-28 -->
<!-- 需求分类：扩展已有算子能力；优先级：P2 -->

# [Operator Request] dlc_kimi_k3_kda_packed_decode（K3 packed QKV KDA decode）

## 需求类型

扩展已有算子能力

**扩展说明**

- 现有 `torch.dlc_kda_recurrent_decode` 已覆盖 K3 recurrent core，当前 native wrapper
  已用它完成真实 decode。
- 现有 `torch.dlc_vllm_recurrent_gated_delta_rule_packed_decode(mixed_qkv, a, b, A_log,
  dt_bias, scale, ssm_state_indices, use_qk_l2norm=False)` 已支持 packed QKV 输入，但 gate
  仍是 GDN softplus 语义。
- K3 需要 bounded sigmoid gate：
  `gate = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))`，并对 raw beta 做 sigmoid。
- 建议扩展 packed decode 入口，增加 `lower_bound` / `gate_mode` 或提供新的 K3 专用 wrapper，
  避免插件侧先 split/reshape q/k/v、构造临时 `cu_seqlens` 后调用通用 recurrent op。

## 优先级

P2

理由：KDA core 已由 DLC recurrent kernel 执行，当前 wrapper 的 split/reshape 成本小于 MoE；
在 #01 MoE 和 MLA epilogue 收敛后再按 decode profile 决定是否启用。它主要减少 18 个 KDA
层的固定 launch/索引开销，并让 packed decode 更容易进入 graph。

## 目标版本与期望时间

vLLM DLC `v0.30.1rc0` / Kimi-K3 decode 优化阶段；日期待算子组排期确认。

## 算子名称

`torch.dlc_kimi_k3_kda_packed_decode`

也可实现为 `dlc_vllm_recurrent_gated_delta_rule_packed_decode` 的扩展参数：
`gate_mode="kda_bounded_sigmoid", lower_bound=-5.0`。本 issue 建议使用 K3 专用公开名，减少
对其它模型 GDN 语义的影响。

## 背景与功能说明

使用场景：

- Kimi-K3 的 18 个 KDA 层（完整模型 69 层）；
- decode 每个 token 先完成 short conv，得到 packed `mixed_qkv`；
- packed 布局为 `[q(H*128) | k(H*128) | v(H*128)]`；
- TP8 每卡 `H=12`，packed width=4608；
- recurrent state 为 FP32 `[num_pages, H, 128, 128]`。

当前问题：

- K3 上游 `fused_recurrent_kda_packed_decode` 是 Triton，DLC 无法执行；
- 当前插件先 split q/k/v、reshape 成 `[1,T,H,D]`，再构造 `arange(T+1)` 作为 cu_seqlens，
  最后调用 `torch.dlc_kda_recurrent_decode`；
- 这些准备操作每层每步重复，且产生额外 view、临时 index tensor 和更多 dispatch。

目标功能：

- 直接消费 packed post-conv QKV；
- 在 kernel 内完成 q/k L2 norm、KDA gate、beta sigmoid、state 更新和输出；
- 消除插件侧 split/reshape/cu_seqlens 构造。
- `out` 应可直接指向 `core_attn_out` 的 `[1,T,H,Dv]` buffer；调用方不再做
  `core_attn_out_non_spec -> core_attn_out` 的整块拷贝。

非目标：

- 不包含 short conv；conv 继续使用 `dlc_kda_causal_conv_update`；
- 不包含 output RMSNorm + sigmoid gate；继续使用
  `dlc_kda_rmsnorm_sigmoid_gate`；
- 不支持 speculative decode / num_accepted_tokens；K3 当前非投机路径；
- 不做 gather/scatter state pool。

## 期望 API 与函数签名

```python
def dlc_kimi_k3_kda_packed_decode(
    mixed_qkv: torch.Tensor,          # [T, 3*H*D], BF16
    raw_g: torch.Tensor,              # [1, T, H, D], BF16
    raw_beta: torch.Tensor,           # [1, T, H], BF16
    A_log: torch.Tensor,              # [H], FP32
    dt_bias: torch.Tensor,            # [H*D], FP32
    initial_state: torch.Tensor,      # [num_pages, H, Dv, Dk], FP32，原位更新
    state_indices: torch.Tensor,      # [T], int32，<=0=null；正值为 state 物理行号
    *,
    lower_bound: float = -5.0,
    scale: float | None = None,
    out: torch.Tensor | None = None,  # [1, T, H, Dv], BF16
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

## 参数与返回值

输入 / 属性：

- `mixed_qkv`：`[T, 2*H*Dk + H*Dv]` BF16；K3 `Dk=Dv=128`，TP8 width=4608。
- `raw_g`：KDA raw gate logits，`[1,T,H,Dk]`。
- `raw_beta`：raw beta logits，`[1,T,H]`；kernel 内 sigmoid。
- `A_log`：每 head 一个 FP32 参数。
- `dt_bias`：每 head 的 K 维 bias，长度 `H*Dk`。
- `initial_state`：分页 recurrent state，最后一个 `Dv×Dk` 维度连续。
- `state_indices`：每个 token 的物理 state page；`<=0` 为 null，输出为 0 且不更新 state。
  正值直接作为 `initial_state[page]` 的 dim0 行号，不做 `page - 1`；K3 的 state row 0
  保留为 null，有效 page 为 `1 .. num_pages-1`。
- `lower_bound`：K3 为 -5.0。
- `scale`：默认 `Dk**-0.5`。

返回值：

- `out`：`[1,T,H,Dv]` BF16；
- `initial_state`：同一个原位更新后的 state tensor。

## 功能语义

对每个 token 和 head：

```text
q = normalize(mixed_qkv[t, 0:H*Dk]) * scale
k = normalize(mixed_qkv[t, H*Dk:2*H*Dk])
v = mixed_qkv[t, 2*H*Dk:2*H*Dk+H*Dv]

if state_indices[t] <= 0:
    out[t] = 0
    state unchanged
    continue

page = state_indices[t]
S = state[page, h]                         # [Dv, Dk]
gate = lower_bound * sigmoid(
    exp(A_log[h]) * (raw_g[t,h,:] + dt_bias[h,:])
)                                           # [Dk]
S = S * exp(gate)[None, :]                  # 沿 Dk 衰减，广播到 Dv 行
beta = sigmoid(raw_beta[t,h])               # scalar
delta = beta * (v - S @ k)                  # [Dv,Dk] @ [Dk] -> [Dv]
S = S + delta[:, None] * k[None, :]         # [Dv,1] * [1,Dk] -> [Dv,Dk]
out[t,h,:] = S @ q                          # [Dv,Dk] @ [Dk] -> [Dv]
state[page, h] = S
```

Q/K normalization：`x / sqrt(sum(x^2) + 1e-6)`。

state 的固定布局和乘法方向：

- `initial_state`：`[num_pages, H, Dv, Dk]`；
- 每个 `(page, head)` 的 state：`[Dv, Dk]`，最后一维连续；
- gate 形状：`[Dk]`，每个 key channel 一个 decay；
- `S @ k`、`S @ q` 都必须左乘 state；不能写成 `S.T @ k` / `S.T @ q`；
- `delta[:, None] * k[None, :]` 是 `[Dv,Dk]` 的 rank-1 更新。

边界：

- `T=0`：返回 `[1,0,H,Dv]`，state 不变；
- `state_indices<=0`：输出 0，state 不变；
- 正 state index 必须在本调用内唯一，这是调用前置条件（precondition）。不同
  token/head 是并行执行的，不允许依赖“按 token 顺序处理”；重复正 index 会形成同一
  state page 的并发读写竞争，属于非法输入。
- 生产算子不得为了检查重复而在调用路径上读取 device `state_indices` 到 host：
  `tolist()` / `item()` / `int(tensor)` 会产生 D2H 和同步，破坏 graph capture 和
  no-host-sync 目标。唯一性应由 state page allocator/scheduler 在 CPU metadata
  生成阶段保证。
- 如需强校验，只能在非 graph 的 debug/validation 路径执行，或对本来就是 CPU 的
  page metadata 校验后再 H2D；主 kernel 和图内路径假定 precondition 已满足。
- 确有“多个 token 顺序更新同一 state”的需求时，应使用带 `query_start_loc` 的通用
  recurrent decode，而不是本 packed decode。
- 越界 index：报错。

## 支持范围与约束

首期精确范围：

- `H` 动态；TP8 生产值 12；
- `Dk=Dv=128`；
- `T` 动态，至少覆盖 1、8、64、256；
- packed QKV/raw_g/raw_beta BF16；state/A_log/dt_bias FP32；
- `state_indices` int32；
- `initial_state.stride()[1:] == (Dv*Dk, Dk, 1)`；dim0 stride 可大于 `H*Dv*Dk`；
- 单卡 DLC；
- eager 必须通过，目标 graph capture；
- 不支持 `num_accepted_tokens` / speculative decode。
- 支持 strided `mixed_qkv` 行/head view；只在最后一维要求 stride=1。

## 精度要求

Golden / reference：

- 以本 issue 末尾的自包含 PyTorch reference 为准；
- upstream Triton 语义也由该 reference 完整表达。

输入分布：

- packed qkv/raw_g N(0,1)，raw_beta N(0,1)；
- state N(0,0.1)，A_log/dt_bias 使用真实 checkpoint 分布；
- state index 覆盖 null、连续、乱序；至少 200 组。

比较：

- output BF16 转 FP32：`atol=2e-2, rtol=2e-2`；
- final state：`atol=1e-3, rtol=1e-3`；
- null index 的 output 全 0，state bit-exact；
- 与通用 `dlc_kda_recurrent_decode` 在 T=1 时数值一致。

中间精度：Q/K norm、gate、state 更新全部 FP32；output 一次 cast BF16。

## 测试范围

- [x] 基本功能与 reference 对齐
- [ ] 多 dtype 精度（首期 BF16/FP32 state）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非 contiguous、特殊 stride 或 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式（每 rank 独立）
- [x] 长时间稳定性
- [x] 性能基准与性能回归

具体测试用例、shape、数据范围和预期结果：

1. `H ∈ {1,12,96}`，`T ∈ {0,1,8,64,256}`，Dk=Dv=128。
2. state dim0 stride 连续和 padded 两种；state index 含 0、负数、乱序。
   正 index 必须唯一；重复正 index 只放在非 graph 的校验器测试中，不作为 kernel
   运行时 D2H 用例。
3. gate 轴与 state 方向探针：构造非对称 `[Dv,Dk]` state，并让 gate 只沿 Dk 变化，
   验证 decay 广播到 Dv 行而不是 Dk 列；同时验证 `S @ k`、`S @ q`，转置实现必须失败。
4. 直接 page 索引探针：令 `state[p]` 与 `state[p-1]` 不同，正 index `p` 必须直接访问
   `state[p]`，不得读取或写回 `state[p-1]`。
5. beta 极值、gate 下界、全 0 q/k、state 全 0。
6. 与通用 recurrent decode 的 T=1 交叉验证。
7. 性能：T=1 单次目标 ≤ 100 us；18 层 decode 总成本 ≤ 2 ms；相对当前 split+recurrent
   wrapper 减少至少一次 host 侧准备和 2 次 dispatch。

## 验证环境与依赖版本

```text
DLC 硬件：8 x TYD，TP8；driver v26.0.1 / Api 26
vLLM：85fd014d5ff4
vLLM DLC（vllm-cl）：ab819da9a0f4 + K3 适配工作区改动
PyTorch / torch-dlc：2.5.0，9049914ec94a
DLC_Custom_Kernel：55b9aef7e0cb
模型：Kimi-K3-INT4；TP8 KDA local H=12，Dk=Dv=128，conv=4，lower_bound=-5
```

## 验收标准

- [ ] API、参数校验和错误行为符合约定。
- [ ] null state index 输出 0 且 state bit-exact。
- [ ] gate 沿 Dk 衰减；state 为 `[Dv,Dk]`，使用 `S @ k` / `S @ q`。
- [ ] 正 state index 直接索引初始 state，不做 `-1`；唯一性作为 precondition，
      graph 路径不产生 D2H；可选非 graph 校验器能拒绝重复正 index。
- [ ] output `atol=2e-2, rtol=2e-2`；state `atol=1e-3, rtol=1e-3`。
- [ ] 与通用 recurrent decode 在 T=1 数值一致。
- [ ] T=1 ≤ 100 us、18 层 ≤ 2 ms（首期目标）。
- [ ] eager 无 host D2H；目标支持 graph capture。
- [ ] 文档、测试和示例已提交。
- [ ] 接入后删除 `core_attn_out[0, :num_actual_tokens] = ...` 拷贝，直接把 `core_attn_out`
      传给 `out`。
- [ ] 后续 KDA gated RMSNorm 使用 `.out(..., out=core_attn_out)`，删除
      `core_attn_out.copy_(o_norm(...))`。

## Reference、日志与附件

### Self-contained PyTorch reference

```python
import torch


def kda_packed_decode_reference(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    lower_bound: float = -5.0,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-token KDA recurrent decode from packed post-conv QKV."""
    tokens = mixed_qkv.shape[0]
    heads = raw_g.shape[2]
    key_dim = raw_g.shape[3]
    value_dim = initial_state.shape[2]
    scale = key_dim**-0.5 if scale is None else scale

    q, k, v = mixed_qkv.split(
        [heads * key_dim, heads * key_dim, heads * value_dim], dim=-1
    )
    q = q.view(tokens, heads, key_dim).float()
    k = k.view(tokens, heads, key_dim).float()
    v = v.view(tokens, heads, value_dim).float()

    q = q / torch.sqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    q = q * scale

    output = torch.zeros(
        (1, tokens, heads, value_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    bias = dt_bias.view(heads, key_dim).float()
    a = torch.exp(A_log.float())
    pages = [int(page) for page in state_indices.detach().cpu().tolist()]
    positive_pages = [page for page in pages if page > 0]
    if len(positive_pages) != len(set(positive_pages)):
        raise ValueError(
            "positive state_indices must be unique within one packed decode call"
        )
    if any(page >= initial_state.shape[0] for page in positive_pages):
        raise ValueError("state index is outside initial_state")

    for token_idx in range(tokens):
        page = pages[token_idx]
        if page <= 0:
            continue

        state = initial_state[page].float()  # [H, Dv, Dk]
        gate = lower_bound * torch.sigmoid(
            a[:, None] * (raw_g[0, token_idx].float() + bias)
        )
        # gate: [H, Dk], state: [H, Dv, Dk] -> broadcast over Dv.
        state = state * torch.exp(gate)[:, None, :]

        beta = torch.sigmoid(raw_beta[0, token_idx].float())
        delta = beta[:, None] * (
            v[token_idx] - torch.einsum("hvk,hk->hv", state, k[token_idx])
        )
        state = state + delta[:, :, None] * k[token_idx][:, None, :]
        initial_state[page] = state

        output[0, token_idx] = torch.einsum(
            "hvk,hk->hv", state, q[token_idx]
        ).to(mixed_qkv.dtype)

    return output, initial_state
```

关键固定输入：

```text
TP8：H=12，Dk=Dv=128，mixed_qkv width=4608
lower_bound=-5.0，scale=128**-0.5
state_indices<=0 表示 null slot：输出全 0，state 不变
正 state index 直接索引 initial_state 的 dim0，不做 page-1
```
