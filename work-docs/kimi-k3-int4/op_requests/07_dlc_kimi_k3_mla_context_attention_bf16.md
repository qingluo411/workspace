<!-- GitHub Issue Form：算子需求字段 -->
<!-- 生成日期：2026-10-28 -->
<!-- 需求分类：新增算子；优先级：P2（V1 可行目标之外的 strict follow-up） -->

# [Operator Request] dlc_kimi_k3_mla_context_attention_bf16（K3 MLA context 直消费分离 K/V）

## 需求类型

新增算子

## 优先级

P2

理由：它不是 V1 可行目标的阻断项。V1 允许 chunked-context 保留 native
`fused_mla_kv_concat` 的一次 `torch.cat`；如果后续要求长 context 也严格零拷贝，单独新增
一个 concat 算子没有有效计算，应该直接做这个融合 context attention，消费
`k_nope/k_pe/v` 并完成 attention。

## 目标版本与期望时间

vLLM DLC `v0.30.1rc0` 之后的 K3 长上下文优化里程碑；日期待算子组排期确认。

## 算子名称

`torch.dlc_kimi_k3_mla_context_attention_bf16`

## 背景与功能说明

使用场景：

- Kimi-K3 MLA 的 chunked-context prefill；
- 每个 context chunk 已从 paged latent cache gather 到 workspace，然后执行 kv_b_proj；
- `k_nope` 与 `v` 是 kv_b_proj 输出的 strided split view；
- `k_pe` 是单独的 RoPE key；
- 当前 `_compute_prefill_context` 先通过 `fused_mla_kv_concat` 把 K 变成连续 full-K，
  再调用 flash attention。

当前问题：

- native `fused_mla_kv_concat` 是 `torch.cat`，是纯数据搬移和形状组合；
- 它不符合“每个算子至少有有效计算，不是纯 copy/reshape”的目标；
- 如果单独提交一个融合 concat kernel，只是把 copy 藏进 kernel，没有解决数据流问题。

目标功能：

- 直接读取 `q/k_nope/k_pe/v`；
- 在 attention kernel 内完成 K 的 NoPE/RoPE 组合；
- 完成 context attention，输出 attention 结果和 LSE；
- 不物化 full-K，不调用 `torch.cat`。

非目标：

- 不处理 new-token prefill；new-token 见 #03；
- 不写 KV cache；context latent 已经 gather 到 workspace；
- 不做 decode attention；decode 见 #04；
- 不做 FP8 / fp8_ds_mla；首期 BF16。

## 期望 API 与函数签名

```python
def dlc_kimi_k3_mla_context_attention_bf16(
    q: torch.Tensor,                 # [T, H, 192], BF16
    k_nope: torch.Tensor,            # [T, H, 128], BF16
    k_pe: torch.Tensor,              # [T, 64] 或 [T, 1, 64], BF16
    v: torch.Tensor,                 # [T, H, 128], BF16，允许 strided/offset view
    cu_seqlens_q: torch.Tensor,      # [B + 1], int32
    cu_seqlens_k: torch.Tensor,      # [B + 1], int32
    seqused_k: torch.Tensor,         # [B], int32
    *,
    max_query_len: int,
    max_seq_len: int,
    scale: float,
    block_table: torch.Tensor | None = None,
    out: torch.Tensor | None = None, # [T, H, 128], BF16
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

## 参数与返回值

输入 / 属性：

- `q`：full query，head dim=192。
- `k_nope`：context key 的 NoPE 部分。
- `k_pe`：已旋转的 RoPE key；按 heads 广播。
- `v`：context value；允许 `[T,H,128]` 的 strided view，尤其要直接接受 kv_b_proj
  split 产生的 offset-V view。
- `cu_seqlens_q` / `cu_seqlens_k` / `seqused_k`：context chunk 的变长信息。
- `max_query_len` / `max_seq_len` / `scale`：attention 参数。
- `block_table`：兼容当前 prefill backend 的接口；context workspace 路径可为空/dummy。

返回值：

- `out`：`[T,H,128]` BF16 attention output；
- `lse`：`[H,T]` FP32 log-sum-exp，供 context partial merge。

## 功能语义

```text
K = broadcast_concat(k_nope, k_pe)  # 不落盘 full-K
for each logical context chunk:
    out = softmax(q @ K^T * scale + causal_mask_if_any) @ V
    lse = logsumexp(scores)
```

行为要求：

- 不调用 `torch.cat` 或等价的独立 K materialize；
- query/key/value 的 K 组合、softmax 和 V matmul 在同一个算子/同一组融合 kernel 中完成；
- context attention 的 `causal=False`（context 后接 new tokens）行为与
  `DLCFlashAttnPrefillBackend.run_prefill_context_chunk` 一致；
- `T == 0` 返回正确空形状；空 chunk 不启动不必要的 kernel；
- 输入不修改；输出和 LSE 可写 caller-provided `out`/workspace。

## 支持范围与约束

首期精确范围：

- `H=12`（TP8），qk=192，nope=128，rope=64，v=128；
- BF16 q/k_nope/k_pe/v 和输出；
- `T` 动态，覆盖 1、64、256、4096；
- `v` 支持 `[T,H,128]` strided/offset view；
- 单卡 DLC；支持 eager，目标支持 breakable graph；
- 不要求 FP8、decode、new-token cache insert。

## 精度要求

Golden / reference：

- 以本 issue 末尾的自包含 PyTorch reference 为准；
- reference 的语义等价于“先 `cat(k_nope,k_pe)`，再做 context attention”；但请求算子
  不允许在 kernel 外实际物化 full-K。

测试：

- `atol=1e-2, rtol=1e-2`（BF16 out 对 FP32 reference）；
- LSE 与 reference 在有限值区域 `atol=1e-2`；
- 空 chunk、单 token chunk、非均匀 `seqused_k`、V offset view。

## 测试范围

- [x] 基本功能与 reference 对齐
- [ ] 多 dtype 精度（首期 BF16）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非 contiguous、特殊 stride 或 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式（每 rank 独立 context）
- [x] 长时间稳定性
- [x] 性能基准与性能回归

具体测试：

1. `T ∈ {0,1,64,256,4096}`，真实 12/192/128/64 几何。
2. `v` 连续与 offset-V view 两种；结果一致。
3. 非均匀 `cu_seqlens`、多 chunk、空 chunk。
4. 与 native cat+attention 对齐。
5. 性能：相对 native cat+attention 减少一次 K materialize，长 context 下目标 ≥ 1.1×。

## 验证环境与依赖版本

```text
DLC 硬件：8 x TYD，TP8；driver v26.0.1 / Api 26
vLLM：85fd014d5ff4 + K3 适配工作区改动
vLLM DLC（vllm-cl）：ab819da9a0f4 + K3 适配工作区改动
PyTorch / torch-dlc：2.5.0，9049914ec94a
DLC_Custom_Kernel：55b9aef7e0cb
模型：Kimi-K3-INT4-Slice8-TP8 / full K3
```

## 验收标准

- [ ] API、参数校验和错误处理符合约定。
- [ ] 不调用独立 `torch.cat`/`fused_mla_kv_concat` 物化 full-K。
- [ ] offset-V view 直接可用，结果与 contiguous V 一致。
- [ ] out/LSE 与 native context backend 对齐，误差满足 `1e-2`。
- [ ] 在 chunked-context 路径无额外整块 K/V copy。
- [ ] eager 与 breakable graph 结果一致。
- [ ] 文档、测试和示例已提交。

## Reference、日志与附件

### Self-contained PyTorch reference

```python
import torch


def mla_context_attention_reference(
    q: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for context attention with separate K-noPE/K-RoPE inputs."""
    del max_query_len, max_seq_len
    tokens, heads, head_dim = q.shape
    value_dim = v.shape[-1]
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    output = torch.zeros(
        (tokens, heads, value_dim), dtype=q.dtype, device=q.device
    )
    lse = torch.full(
        (heads, tokens), float("-inf"), dtype=torch.float32, device=q.device
    )

    batch = seqused_k.shape[0]
    for batch_idx in range(batch):
        q_start = int(cu_seqlens_q[batch_idx])
        q_end = int(cu_seqlens_q[batch_idx + 1])
        k_start = int(cu_seqlens_k[batch_idx])
        k_end = int(cu_seqlens_k[batch_idx + 1])
        kv_len = int(seqused_k[batch_idx])
        if q_start == q_end:
            continue

        q_cur = q[q_start:q_end].float()
        # This cat is only in the semantic reference. The requested operator
        # must assemble K inside its attention kernel and never materialize it.
        k_cur = torch.cat(
            (
                k_nope[k_start:k_end],
                k_pe[k_start:k_end]
                .unsqueeze(1)
                .expand_as(k_nope[k_start:k_end]),
            ),
            dim=-1,
        )
        k_cur = k_cur[:kv_len].float()
        v_cur = v[k_start:k_end][:kv_len].float()

        scores = torch.einsum("qhd,khd->hqk", q_cur, k_cur) * scale
        logits = torch.logsumexp(scores, dim=-1) if kv_len else torch.full_like(
            scores[..., 0], float("-inf")
        )
        probs = torch.softmax(scores, dim=-1) if kv_len else scores
        out = torch.einsum("hqk,khv->qhv", probs, v_cur)
        output[q_start:q_end] = out.to(q.dtype)
        lse[:, q_start:q_end] = logits

    return output, lse
```

关键几何：

```text
TP8：H=12，q=192，k_nope=128，k_pe=64，v=128
context attention causal=False
V1 允许保留一次 native cat；本算子实现后应删除该 cat
```
