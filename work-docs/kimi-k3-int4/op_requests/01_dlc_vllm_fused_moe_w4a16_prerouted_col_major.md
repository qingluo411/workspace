<!-- GitHub Issue Form：算子需求字段 -->
<!-- 生成日期：2026-10-28 -->
<!-- 需求分类：扩展已有算子能力；优先级：P0 -->

# [Operator Request] dlc_vllm_fused_moe_w4a16_prerouted_col_major（K3 W4A16-G128 pre-routed MoE + SiTU）

## 需求类型

扩展已有算子能力

**扩展说明**

- 现有底层能力：`torch.dlc_vllm_fused_moe_w4a16_awq_col_major` 及
  `custom_vllm_fused_moe_w4a16_sym_col_major_g*` 系列已经覆盖 symmetric INT4、
  group scale、col-major 权重和 selected expert 的矩阵计算。
- 现有缺口：
  1. vLLM 在进入量化方法前已经完成 top-k，K3 传入的是 `topk_weights/topk_ids`；
     现有 API 仍接收完整 `gating_output` 并在 kernel 内重新路由。
  2. K3 使用 compressed-tensors 的直接 INT4 pack 顺序，不是 AWQ 的 reverse-order。
  3. K3 routed expert activation 是 SiTU-GLU（beta=4，linear_beta=25），现有内核只支持
     silu/swiglu-limit。
  4. K3 routed expert 的输入宽度是 latent 3584，不是 hidden 7168。
- 落地方式：新增 pre-routed 入口；路由表直接消费 `topk_ids/weights`，底层复用现有
  symmetric col-major G128 子内核，并增加 compressed-tensors 解包和 SiTU activation。

## 优先级

P0

理由：这是当前唯一阻断性能验收的算子。真实 24 层 TP8 切片在 native fallback 上
1-token 请求约 25–42 s，10-token prompt 约 103 s；profile 128-token 前向会超过
DLC 600 秒队列超时，只能在 profile 阶段临时返回零输出。没有该算子，完整 K3 无法做
可用性、长 context 或 TP32 性能验收。

## 目标版本与期望时间

vLLM DLC `v0.30.1rc0` / Kimi-K3 单机 TP8 性能里程碑；具体交付日期待算子组排期确认。

## 算子名称

`torch.dlc_vllm_fused_moe_w4a16_prerouted_col_major`

建议同时保留现有 `dlc_vllm_fused_moe_w4a16_awq_col_major` 不变，避免破坏 GLM/其它模型。

## 背景与功能说明

使用场景：

- Kimi-K3-INT4 的 92 个 MoE 层（当前 24 层切片中为 23 层）；
- 每层 896 routed experts，每 token 选择 16 个专家；
- latent MoE 输入宽度 3584；
- expert intermediate size 3072，TP8 每 rank 本地 intermediate size 384；
- 权重为 compressed-tensors `pack-quantized`、INT4 symmetric、group size 128、W4A16；
- 输入/输出/scale 为 BF16，权重为 int32 packed。

当前问题：

- native fallback 读取 `topk_weights/topk_ids` 后逐 token、逐 expert：
  解包 INT4、反量化、做 gate/up、SiTU、down projection，再按 route 累加；
- 24 层真实切片一次 1-token 前向需要 25–42 s，10-token prompt 需要 102.83 s；
- 该路径产生大量 Python 循环、DLC 小算子和中间 BF16 张量，无法进入 graph。

目标功能：

- 在已有“专家已选好”的前提下，用少量 DLC kernel 完成 selected expert 的
  W4A16 GEMM、SiTU-GLU、down projection 和 route 加权累加；
- 不执行 router / top-k / renormalize；
- 不修改现有 non-prerouted API 的行为。

非目标：

- 不计算 router logits、group top-k、e_score_correction_bias 或 routed scaling；
- 不包含 shared experts；
- 不包含 latent down projection、routed expert RMSNorm 和 up projection；
- 不包含 TP all-reduce；调用方仍按 vLLM FusedMoE 现有 reduction 流程执行。

## 期望 API 与函数签名

```python
def dlc_vllm_fused_moe_w4a16_prerouted_col_major(
    hidden_states: torch.Tensor,       # [T, hidden], BF16
    w13_qweight: torch.Tensor,         # [E, hidden // 8, 2 * I_tp], int32
    w2_qweight: torch.Tensor,          # [E, I_tp // 8, hidden], int32
    w13_scales: torch.Tensor,          # [E, hidden // 128, 2 * I_tp], BF16
    w2_scales: torch.Tensor,           # [E, I_tp // 128, hidden], BF16
    topk_weights: torch.Tensor,        # [T, K], BF16/FP32，路由已归一化/缩放
    topk_ids: torch.Tensor,            # [T, K], int32/int64，-1 表示跳过
    *,
    expert_map: torch.Tensor | None = None,  # [E] int32/int64，global->local，-1 跳过
    activation: str = "situ",
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
    out: torch.Tensor | None = None,   # [T, hidden], BF16
) -> torch.Tensor:
    ...
```

## 参数与返回值

输入：

- `hidden_states`：`[T, hidden]` BF16，latent MoE 输入；K3 为 `hidden=3584`。
- `w13_qweight`：`[E, hidden/8, 2*I_tp]` int32。`I_tp` 是 TP 本地 intermediate size；
  K3 TP8 为 `384`。前 `I_tp` 列为 gate（w1），后 `I_tp` 列为 up（w3）。
- `w2_qweight`：`[E, I_tp/8, hidden]` int32。
- `w13_scales` / `w2_scales`：BF16，分别沿 gate/up 输出维和 hidden 输出维广播；
  每 128 个输入元素一个 scale。
- `topk_weights`：`[T, K]`；已是最终路由权重，算子不再 renormalize。
- `topk_ids`：`[T, K]`；全局 expert id；`-1` 表示该 route 无效。
- `expert_map`：可选；`expert_map[global_id]` 为本地 expert id，`-1` 表示该 rank 不持有。

以上 top-k 和 expert map 必须是 device tensor；kernel/wrapper 禁止 `.cpu()`、`.item()`、
`.tolist()` 或 numpy 转换。

属性：

- `activation`：首期只要求 `"situ"`；未知值必须报错。
- `situ_beta`：K3 为 4.0。
- `situ_linear_beta`：K3 为 25.0；允许 `None` 表示对 up 不做 tanh 缩放。

输出：

- 返回 `[T, hidden]` BF16；若提供 `out`，原位写入并返回同一个 tensor。

行为：

- 不修改 `hidden_states`、权重、scale、top-k 或 expert map；
- 输出是当前 TP rank 的 partial down-projection，调用方负责后续 all-reduce。

## 功能语义

### 5.1 compressed-tensors INT4 解包

packed weight 是 compressed-tensors 的标准 int4 pack：每个 int32 从低位到高位包含 8 个
4-bit 值，**不使用 AWQ reverse-order**。解包后用 unsigned offset 转 signed：

```text
q = (packed_int32 >> (4 * lane)) & 0xF       lane = 0..7
w = (q - 8).float() * scale
```

`w13` 的 gate/up 排列：

```text
gate_weight = dequant(w13_qweight[..., :I_tp])
up_weight   = dequant(w13_qweight[..., I_tp:])
```

### 5.2 selected expert 计算

```text
for token t in 0..T-1:
    acc = zeros(hidden, fp32)
    for route r in 0..K-1:
        global_e = topk_ids[t, r]
        if global_e < 0: continue
        local_e = expert_map[global_e] if expert_map is not None else global_e
        if local_e < 0: continue

        gate = hidden_states[t] @ dequant(w13_gate[local_e]).T
        up   = hidden_states[t] @ dequant(w13_up[local_e]).T

        gate_act = situ_beta * tanh(gate / situ_beta) * sigmoid(gate)
        if situ_linear_beta is not None:
            up = situ_linear_beta * tanh(up / situ_linear_beta)

        y = (gate_act * up) @ dequant(w2[local_e]).T
        acc += y * topk_weights[t, r]
    out[t] = acc.to(BF16)
```

实现可以使用 grouped GEMM：先按 expert 聚合 token，再做一次 selected-expert GEMM，
最后 unpermute 并按 `topk_weights` 累加；数学语义与上式一致。

### 5.3 边界与异常

- `T == 0`：返回形状正确的空 tensor；
- `topk_ids=-1` 或 `expert_map=-1`：跳过 route，不报错；
- 相同 expert 在同一 token 的多个 route 中出现：按多个 route 分别累加；
- `topk_ids` 越界、形状不匹配、packed/scale shape 不匹配：报错；
- 不做任何 router softmax、top-k 选择或权重归一化；
- 不使用随机数，相同输入必须确定。

## 支持范围与约束

首期 P0 精确范围：

- `hidden = 3584`；
- `E = 896`；
- `K = 16`；
- `I_tp = 384`（TP8 全模型 routed expert local intermediate）；
- `group_size = 128`，symmetric INT4，无 zero point；
- `hidden_states`/输出/scales：BF16；
- packed weight：int32；
- `topk_weights`：BF16 或 FP32，`topk_ids`：int32 或 int64；
- 单卡 DLC；不包含 TP collective；
- 支持 eager，目标支持 DLC Graph capture/replay。
- 内部 grouped-token scratch 必须由持久 workspace/预分配 buffer 提供，不能在每次调用中做
  依赖数据的 host 分配。

通用范围建议：

- `T ∈ [0, max_num_batched_tokens]` 动态；
- `E`、`K` 动态；
- `hidden`、`I_tp` 为 128 的倍数，且 packed 维度能被 int32 完整装下；
- `expert_map` 首期只要求“全 local”和“TP/EP 本地子集”两种；
- 非 contiguous 输入不要求；要求最后一维 stride=1。

禁止/暂不支持：

- AWQ reverse-order pack、非对称 zero point、G32/G64（可复用现有其它入口）；
- bias、activation 以外的 epilogue、shared expert、latent norm/up projection；
- 算子内跨 rank all-reduce。

## 精度要求

Golden / reference 实现：

- 以本 issue 末尾的自包含 PyTorch reference 为准；
- 解包逻辑也直接内联在 reference 中，不依赖其它源码文件。

输入数据分布：

- `hidden_states ~ N(0, 1)` 和 `~ U(-1, 1)`；
- weight scale 使用真实 checkpoint 的 scale 分布和全 1 两种；
- top-k ids 使用真实 gate 选择、随机选择、含 `-1` 和重复 expert；
- 至少 200 组输入，固定 `torch.manual_seed`。

比较方式：

- 同一 packed 权重、scale、top-k 和权重下，DLC 输出与 native FP32 累加输出比较；
- 比较前统一转 FP32；
- 首期验收：`atol=5e-2`、`rtol=5e-2`；
- 对真实 checkpoint 单 expert 反量化参考要求 bit-exact（当前 native 已验证）。

必须逐元素一致的行为：

- `topk_ids=-1` / `expert_map=-1` 的 route 必须完全不参与累加；
- 相同 route 的 top-k weight 乘法顺序不影响结果；
- 空 token 必须返回空 tensor；
- `out` 提供时必须原位写入，不得另分配返回。

累加精度：gate/up/down GEMM 和 route 累加使用 FP32；最终一次 cast 到 BF16。

## 测试范围

- [x] 基本功能与 reference 对齐
- [ ] 多 dtype 精度（首期只要求 BF16 + int32 packed）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非 contiguous、特殊 stride 或 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [x] 多卡/分布式（只验证 expert_map/TP 本地 partial，不验证 collective）
- [ ] 长时间稳定性
- [x] 性能基准与性能回归

具体测试用例、shape、数据范围和预期结果：

1. 功能：`T ∈ {1,8,64,256,4096}`，`E=896`，`K=16`，`hidden=3584`，`I_tp=384`；
   与 native reference 对齐。
2. 边界：`T=0`；所有 `topk_ids=-1`；部分 route 无效；重复 expert；全 0 weight/scale。
3. `expert_map`：全 local、稀疏 local map、含 -1。
4. layout：真实 checkpoint packed layout；确认不能套用 AWQ reverse-order。
5. 异常：越界 expert id、packed dim 不匹配、scale group 不匹配、未知 activation。
6. 性能：
   - 单次 `T=1`、K=16 的 selected-expert 计算目标 ≤ 4 ms/层；
   - 24 层切片 `T=1` 的 MoE 部分目标 ≤ 100 ms，解锁 1 token 秒级请求；
   - `T=64` 的 24 层 MoE 部分目标 ≤ 1 s；
   - 不允许 host D2H/sync；目标支持 Graph replay。

## 验证环境与依赖版本

```text
DLC 硬件：8 x TYD，/dev/cltech0..7，driver v26.0.1 / Api 26
vLLM：85fd014d5ff4（release-kimi-k3-int4 + 本次 K3 适配工作区改动）
vLLM DLC（vllm-cl）：ab819da9a0f4（+ K3 适配工作区改动）
PyTorch / torch-dlc：2.5.0，9049914ec94a
DLCSynapse：fe4f23a783ad
DLC_Custom_Kernel：55b9aef7e0cb
LLVM：baef94cacd90
模型：/data/models/Kimi-K3-INT4-Slice8-TP8；full K3 为 93 层、92 个 MoE 层
关键环境变量：
  VLLM_PLUGINS=dlc,dlc_patches
  CUSTOM_KERNEL_PATH=/usr/local/chipltech/synapse/lib
```

## 验收标准

- [ ] API、参数校验、空 tensor 和错误行为符合约定。
- [ ] compressed-tensors INT4 pack 顺序正确，不能错误套用 AWQ reverse-order。
- [ ] `topk_ids=-1`、`expert_map=-1` 和重复 route 行为与 reference 一致。
- [ ] 不执行 router / top-k / renormalize，top-k weights 只作为最终乘法系数。
- [ ] hidden_states 与输出都为 `[T,3584]`，模型编排边界不插入 token reshape/permute。
- [ ] 全路径无 `.cpu()`、`.item()`、`.tolist()`、`.numpy()`；top-k 和 expert map 保持 device tensor。
- [ ] 真实 checkpoint 单 expert 反量化与 reference bit-exact。
- [ ] 真实 W4A16-G128 权重在 `T ∈ {1,64,256}` 精度满足 `atol=5e-2, rtol=5e-2`。
- [ ] 24 层切片 `T=1` 的 MoE 时间 ≤ 100 ms，`T=64` ≤ 1 s（首期目标）。
- [ ] eager 下无 host D2H/sync；目标支持 DLC Graph capture/replay。
- [ ] 文档、测试、reference 和接入示例已提交。

## Reference、日志与附件

### Self-contained PyTorch reference

```python
import torch


def unpack_int4_from_int32_dim0(
    packed: torch.Tensor,
    output_rows: int,
) -> torch.Tensor:
    """Unpack compressed-tensors int4 layout (packed along dimension 0)."""
    assert packed.dtype == torch.int32
    pack_factor = 8
    mask = 0xF
    unpacked = torch.zeros(
        (packed.shape[0] * pack_factor, packed.shape[1]),
        dtype=torch.int32,
        device=packed.device,
    )
    for lane in range(pack_factor):
        unpacked[lane::pack_factor, :] = (
            packed >> (4 * lane)
        ) & mask
    unpacked = unpacked[:output_rows]
    return (unpacked - 8).to(torch.int8)


def dequantize_packed_weight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    shape: tuple[int, int],
    group_size: int,
) -> torch.Tensor:
    unpacked = unpack_int4_from_int32_dim0(packed, shape[0]).float()
    expanded_scale = scale.float().repeat_interleave(group_size, dim=0)
    expanded_scale = expanded_scale[: shape[0]]
    return (unpacked * expanded_scale).to(torch.bfloat16)


def situ_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
) -> torch.Tensor:
    gate_f = gate.float()
    up_f = up.float()
    gate_f = beta * torch.tanh(gate_f / beta) * torch.sigmoid(gate_f)
    if linear_beta is not None:
        up_f = linear_beta * torch.tanh(up_f / linear_beta)
    return (gate_f * up_f).to(gate.dtype)


def fused_moe_w4a16_prerouted_reference(
    hidden_states: torch.Tensor,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """Reference for one rank's local expert shard."""
    tokens, hidden = hidden_states.shape
    intermediate = w2_qweight.shape[1] * 8
    output = torch.zeros(
        (tokens, hidden), dtype=torch.float32, device=hidden_states.device
    )

    for token_idx in range(tokens):
        token = hidden_states[token_idx : token_idx + 1]
        for route_idx in range(topk_ids.shape[1]):
            global_expert = int(topk_ids[token_idx, route_idx])
            if global_expert < 0:
                continue
            expert_id = global_expert
            if expert_map is not None:
                expert_id = int(expert_map[global_expert])
                if expert_id < 0:
                    continue

            w13 = dequantize_packed_weight(
                w13_qweight[expert_id],
                w13_scales[expert_id],
                (hidden, 2 * intermediate),
                128,
            )
            w2 = dequantize_packed_weight(
                w2_qweight[expert_id],
                w2_scales[expert_id],
                (intermediate, hidden),
                128,
            )
            gate_up = torch.mm(token, w13)
            gate, up = gate_up.split(intermediate, dim=-1)
            activated = situ_glu(gate, up, situ_beta, situ_linear_beta)
            down = torch.mm(activated.to(token.dtype), w2)
            output[token_idx].add_(
                down[0].float() * topk_weights[token_idx, route_idx].float()
            )
    return output.to(hidden_states.dtype).view_as(hidden_states)
```

真实权重样例（layer 1 expert 0）：

```text
w1.weight_packed = (3072, 448) int32
w1.weight_scale  = (3072, 28) bf16
w2.weight_packed = (3584, 384) int32
w2.weight_scale  = (3584, 24) bf16
推理期 vLLM 参数布局：
w13 = [E, 448, 768] int32；w2 = [E, 48, 3584] int32
```

实测结果已在上文性能基线与测试目标中给出，不再依赖外部报告文件。
