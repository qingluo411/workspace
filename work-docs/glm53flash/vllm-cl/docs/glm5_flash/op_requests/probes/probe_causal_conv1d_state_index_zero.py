"""#02 探针（2026-09-22 修正版）：确认 ``_causal_conv1d_update`` 对 page 0（NULL_BLOCK_ID）
与 page -1（PAD_SLOT_ID）的跳过语义。

vLLM/DLC 约定（见 vllm/v1/attention/backends/utils.py:46、vllm/v1/core/block_pool.py:188-191、
DLC_Custom_Kernel/dlc_kernels/conv/causal_conv1d_update.c:65-66,145,293,346）:
  * ``conv_state_indices`` 中 0（null block）或 -1（pad slot）的行 -> 输出全零、conv_state 不变；
  * 其余页 -> 输出与 fp32 reference 一致，且 conv_state 滚动为「最近 K-1 个 token」。

2026-09-21 的初版探针把 page 0 当成普通页（拿它的历史当真值），因此误报「page 0 输出错误」。
"""
import torch

dev = 'dlc'
torch.manual_seed(0)
DIM, W, NP = 256, 4, 6
NULL_BLOCK_ID, PAD_SLOT_ID = 0, -1


def native1(x1, w2d, bias, hist):
    """x1: [dim, 1], hist: [dim, W-1] -> single-token silu(depthwise conv) 输出。"""
    win = torch.cat([hist, x1], 1)
    y = (win.float() * w2d.float()).sum(-1) + bias.float()
    return torch.nn.functional.silu(y).to(x1.dtype)


def roll(hist, x1):
    """一步之后的 conv_state（滑窗左移 + 追加新 token）。"""
    return torch.cat([hist[:, 1:], x1], 1)


w2d = torch.randn(DIM, W, dtype=torch.bfloat16, device=dev) * 0.3
bias = torch.randn(DIM, dtype=torch.bfloat16, device=dev) * 0.1
hists = {p: torch.randn(DIM, W - 1, dtype=torch.bfloat16, device=dev) * 0.5 for p in range(NP)}

CASES = [[0], [1], [2], [3], [0, 2], [1, 0], [-1], [-1, 1]]

for idx in CASES:
    buf = torch.zeros(NP, W - 1, DIM, dtype=torch.bfloat16, device=dev)
    for p in {i for i in idx if i >= 0}:
        buf[p].copy_(hists[p].t())
    st = buf.permute(0, 2, 1)                     # [NP, DIM, W-1]，通道维 stride == 1
    st_before = st.clone()
    B = len(idx)
    xu = torch.randn(B, DIM, dtype=torch.bfloat16, device=dev)
    o = torch._causal_conv1d_update(
        xu, w2d, bias, st,
        torch.tensor(idx, dtype=torch.int32, device=dev),
        None, None, -1, None, None, True, True)
    torch.dlc.synchronize()

    parts = []
    for i, p in enumerate(idx):
        out_zero = float(o[i].abs().max().item()) == 0.0
        if p in (NULL_BLOCK_ID, PAD_SLOT_ID):
            if p == NULL_BLOCK_ID:
                parts.append(f"[{p}] skipped out_zero={out_zero} state_unchanged={torch.equal(st[p], st_before[p])}")
            else:                                  # -1 是 pad slot，没有对应页可检查
                parts.append(f"[{p}] skipped out_zero={out_zero}")
        else:
            ref = native1(xu[i].unsqueeze(1), w2d, bias, hists[p])
            ref_state = roll(hists[p], xu[i].unsqueeze(1))
            out_err = (o[i].float() - ref.float()).abs().max().item()
            st_err = (st[p].float() - ref_state.float()).abs().max().item()
            parts.append(f"[{p}] out_err={out_err:.2e} state_err={st_err:.2e}")
    print(f"conv_state_indices={idx} -> " + "; ".join(parts))
