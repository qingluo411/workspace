#!/usr/bin/env python3
"""Probe DLC KDA kernels with Kimi-K3 head geometry.

The GLM-5.3 kernels were validated with TP8 local head counts such as H=8.
Kimi-K3 uses H=96 before tensor parallelism, with Dk=Dv=128.  This probe checks
whether the existing recurrent-decode and chunk-prefill kernels accept that
geometry without changing model code.

Run with an external timeout because an unsupported kernel shape may hang a DLC
queue:

    timeout 75 .venv/bin/python work-tests/kimi_k3/probe_kda_shapes.py \
      --device dlc:5 --mode decode --heads 96 --tokens 1

    timeout 75 .venv/bin/python work-tests/kimi_k3/probe_kda_shapes.py \
      --device dlc:5 --mode chunk --heads 96 --tokens 16 --chunk-size 16
"""

import argparse
import math

import torch


HEAD_DIM = 128
LOWER_BOUND = -5.0


def _decode_reference(q, k, v, raw_g, beta, a_log, g_bias, state):
    q = q.detach().float().cpu()
    k = k.detach().float().cpu()
    v = v.detach().float().cpu()
    raw_g = raw_g.detach().float().cpu()
    beta = beta.detach().float().cpu()
    a_log = a_log.detach().float().cpu()
    g_bias = g_bias.detach().float().cpu()
    state = state.detach().float().cpu()

    gate = LOWER_BOUND * torch.sigmoid(
        torch.exp(a_log)[:, None] * (raw_g[0, 0] + g_bias)
    )
    decayed = state * torch.exp(gate)[:, None, :]
    projected = torch.einsum("hvk,hk->hv", decayed, k[0, 0])
    delta = torch.sigmoid(beta[0, 0])[:, None] * (v[0, 0] - projected)
    next_state = decayed + torch.einsum("hk,hv->hvk", k[0, 0], delta)
    output = torch.einsum("hvk,hk->hv", next_state, q[0, 0]) / math.sqrt(HEAD_DIM)
    return output, next_state


def probe_decode(device, heads):
    torch.manual_seed(20261028)
    shape = (1, 1, heads, HEAD_DIM)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
    k = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
    v = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
    raw_g = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
    beta = torch.randn((1, 1, heads), device=device, dtype=torch.bfloat16) * 0.1
    a_log = torch.randn(heads, device=device, dtype=torch.float32) * 0.1
    g_bias = torch.randn((heads, HEAD_DIM), device=device, dtype=torch.float32) * 0.1
    initial = torch.randn((2, heads, HEAD_DIM, HEAD_DIM), device=device, dtype=torch.float32) * 0.02
    state = initial.clone()
    out = torch.empty((1, 1, heads, HEAD_DIM), device=device, dtype=torch.bfloat16)

    torch.dlc_kda_recurrent_decode(
        q,
        k,
        v,
        raw_g,
        beta,
        a_log,
        g_bias,
        state,
        torch.tensor([1], device=device, dtype=torch.int32),
        torch.tensor([0, 1], device=device, dtype=torch.int32),
        None,
        lower_bound=LOWER_BOUND,
        scale=1.0 / math.sqrt(HEAD_DIM),
        out=out,
    )
    torch.dlc.synchronize()

    ref_out, ref_state = _decode_reference(q, k, v, raw_g, beta, a_log, g_bias, initial[1])
    out_error = (out[0, 0].detach().float().cpu() - ref_out).abs().max().item()
    state_error = (state[1].detach().float().cpu() - ref_state).abs().max().item()
    print(
        f"decode H={heads}: out_err={out_error:.4e} state_err={state_error:.4e}",
        flush=True,
    )
    if out_error > 2e-2 or state_error > 2e-2:
        raise RuntimeError("KDA recurrent-decode numerical check failed")


def probe_chunk(device, heads, tokens, chunk_size):
    torch.manual_seed(20261028)
    shape = (1, tokens, heads, HEAD_DIM)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.2
    k = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.2
    v = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.2
    raw_g = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.2
    beta = torch.randn((1, tokens, heads), device=device, dtype=torch.bfloat16) * 0.2
    a_log = torch.randn(heads, device=device, dtype=torch.float32) * 0.1
    g_bias = torch.randn((heads, HEAD_DIM), device=device, dtype=torch.float32) * 0.1
    initial = torch.randn((1, heads, HEAD_DIM, HEAD_DIM), device=device, dtype=torch.float32) * 0.02
    query_start_loc = torch.tensor([0, tokens], device=device, dtype=torch.int32)

    out, final_state = torch.ops.aten.dlc_kda_chunk_prefill(
        q,
        k,
        v,
        raw_g,
        beta,
        a_log,
        g_bias,
        initial,
        query_start_loc,
        LOWER_BOUND,
        1.0 / math.sqrt(HEAD_DIM),
        chunk_size,
    )
    torch.dlc.synchronize()
    finite = torch.isfinite(out).all().item() and torch.isfinite(final_state).all().item()
    print(
        f"chunk H={heads} N={tokens} C={chunk_size}: "
        f"finite={finite} out_absmean={out.float().abs().mean().item():.4e}",
        flush=True,
    )
    if not finite:
        raise RuntimeError("KDA chunk-prefill produced non-finite output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="dlc:0")
    parser.add_argument("--mode", choices=("decode", "chunk", "both"), default="both")
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()

    torch.dlc.set_device(torch.device(args.device))
    if args.mode in ("decode", "both"):
        probe_decode(args.device, args.heads)
    if args.mode in ("chunk", "both"):
        probe_chunk(args.device, args.heads, args.tokens, args.chunk_size)


if __name__ == "__main__":
    main()
