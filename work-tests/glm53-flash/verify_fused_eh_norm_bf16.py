"""Validate torch.dlc_fused_eh_norm_bf16 against the issue requirements.

Usage:
    source /root/workspace/.venv/bin/activate
    python work-tests/glm53-flash/verify_fused_eh_norm_bf16.py --device 0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch


EPS = 1e-5
RESULTS = Path("/root/workspace/work-tests/glm53-flash/results")


def reference(positions, embeds, previous, enorm_w, hnorm_w, eps=EPS):
    def norm(x, weight):
        xf = x.float()
        rr = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
        return (xf * rr * weight.float()).to(torch.bfloat16)

    masked = embeds * (positions != 0).unsqueeze(-1).to(embeds.dtype)
    return torch.cat([norm(masked, enorm_w), norm(previous, hnorm_w)], dim=-1)


def get_op():
    fn = getattr(torch, "dlc_fused_eh_norm_bf16", None)
    if fn is not None:
        return fn
    return torch.ops.aten.dlc_fused_eh_norm_bf16


def make_positions(tokens, pattern, device):
    if pattern == "all_zero":
        return torch.zeros(tokens, dtype=torch.int64, device=device)
    if pattern == "all_nonzero":
        return torch.arange(1, tokens + 1, dtype=torch.int64, device=device)
    return torch.tensor(
        [0 if i % 3 == 0 else i + 1 for i in range(tokens)],
        dtype=torch.int64,
        device=device,
    )


def make_matrix(tokens, hidden, row_stride, device):
    base = torch.empty((tokens, row_stride), dtype=torch.bfloat16, device=device)
    if row_stride != hidden:
        base.normal_()
        base[:, hidden:] = 0
        return base[:, :hidden]
    base.normal_()
    return base


def check_case(op, device, tokens, hidden, pattern, row_stride, seed):
    torch.manual_seed(seed)
    cpu_e = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    cpu_p = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    cpu_ew = torch.randn(hidden, dtype=torch.bfloat16)
    cpu_hw = torch.randn(hidden, dtype=torch.bfloat16)
    pos = make_positions(tokens, pattern, device)
    e = make_matrix(tokens, hidden, row_stride, device)
    p = make_matrix(tokens, hidden, row_stride, device)
    e.copy_(cpu_e.to(device))
    p.copy_(cpu_p.to(device))
    ew = cpu_ew.to(device)
    hw = cpu_hw.to(device)
    expected = reference(
        torch.tensor([0 if i % 3 == 0 else i + 1 for i in range(tokens)], dtype=torch.int64)
        if pattern == "mixed"
        else (torch.zeros(tokens, dtype=torch.int64) if pattern == "all_zero" else torch.arange(1, tokens + 1, dtype=torch.int64)),
        cpu_e,
        cpu_p,
        cpu_ew,
        cpu_hw,
        EPS,
    )
    if pattern == "mixed":
        pos_cpu = torch.tensor([0 if i % 3 == 0 else i + 1 for i in range(tokens)], dtype=torch.int64)
    elif pattern == "all_zero":
        pos_cpu = torch.zeros(tokens, dtype=torch.int64)
    else:
        pos_cpu = torch.arange(1, tokens + 1, dtype=torch.int64)
    got = op(pos, e, p, ew, hw, EPS)
    torch.dlc.synchronize()
    if tokens == 0:
        assert got.shape == (0, 2 * hidden)
        return
    torch.testing.assert_close(got.cpu(), expected, atol=2e-2, rtol=2e-2)
    assert torch.isfinite(got).all().item()
    zero_rows = pos_cpu == 0
    if zero_rows.any():
        assert torch.count_nonzero(got.cpu()[zero_rows, :hidden]).item() == 0


def bench(fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn()
    torch.dlc.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.dlc.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    device = torch.device(f"dlc:{args.device}")
    torch.dlc.set_device(device)
    op = get_op()
    results = {"device": str(device), "correctness": [], "layout": [], "graph": {}, "perf": {}}

    for hidden in (1024, 2048, 4096):
        for tokens in (0, 1, 2, 4, 8, 16, 32, 40, 64, 128):
            for pattern in ("all_zero", "all_nonzero", "mixed"):
                check_case(op, device, tokens, hidden, pattern, hidden, seed=hidden + tokens + len(pattern))
                results["correctness"].append({"tokens": tokens, "hidden": hidden, "pattern": pattern, "ok": True})

    for hidden in (1024, 4096):
        for tokens in (4, 40):
            for row_stride in (hidden, hidden + 1, 2 * hidden):
                check_case(op, device, tokens, hidden, "mixed", row_stride, seed=9000 + hidden + tokens + row_stride)
                results["layout"].append({"tokens": tokens, "hidden": hidden, "row_stride": row_stride, "ok": True})

    for seed in range(200):
        check_case(op, device, 40, 4096, "mixed", 4096, seed=10000 + seed)
    results["random_200"] = "PASS"

    # Error paths required by the issue.
    tokens, hidden = 4, 4096
    e = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    p = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    w = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    pos = torch.arange(tokens, dtype=torch.int64, device=device)
    e_bad_stride_base = torch.empty((tokens, hidden * 2), dtype=torch.bfloat16, device=device)
    e_bad_stride = e_bad_stride_base[:, ::2]
    errors = {}
    for name, fn in {
        "positions_length": lambda: op(pos[:3], e, p, w, w, EPS),
        "shape_mismatch": lambda: op(pos, e, p[:3], w, w, EPS),
        "weight_length": lambda: op(pos, e, p, w[:128], w, EPS),
        "last_stride": lambda: op(pos, e_bad_stride, p, w, w, EPS),
        "dtype": lambda: op(pos, e.float(), p, w, w, EPS),
        "nonfinite_eps": lambda: op(pos, e, p, w, w, math.inf),
    }.items():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            errors[name] = type(exc).__name__
        else:
            raise AssertionError(f"expected error for {name}")
    results["errors"] = errors

    # Graph capture/replay: fixed shapes, 1000 replays.
    tokens, hidden = 40, 4096
    e = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    p = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    ew = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    hw = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    pos = make_positions(tokens, "mixed", device)
    graph = torch.dlc.DLCGraph()
    with torch.dlc.graph(graph):
        out = op(pos, e, p, ew, hw, EPS)
    ptr = out.data_ptr()
    expected = reference(pos.cpu(), e.cpu(), p.cpu(), ew.cpu(), hw.cpu(), EPS)
    for _ in range(1000):
        graph.replay()
        torch.dlc.synchronize()
        assert out.data_ptr() == ptr
        torch.testing.assert_close(out.cpu(), expected, atol=2e-2, rtol=2e-2)
    results["graph"] = {"tokens": tokens, "hidden": hidden, "replays": 1000, "ok": True}

    # Performance: fused op vs the Python fallback for MTP-relevant shapes.
    def fallback(positions, embeds, previous):
        mask = (positions != 0).unsqueeze(-1)
        masked = embeds * mask.to(embeds.dtype)
        return torch.cat(
            [
                (masked.float() * torch.rsqrt(masked.float().square().mean(-1, keepdim=True) + EPS) * ew.float()).to(torch.bfloat16),
                (previous.float() * torch.rsqrt(previous.float().square().mean(-1, keepdim=True) + EPS) * hw.float()).to(torch.bfloat16),
            ],
            dim=-1,
        )

    for tokens in (4, 8, 16, 40):
        e = torch.randn(tokens, 4096, dtype=torch.bfloat16, device=device)
        p = torch.randn(tokens, 4096, dtype=torch.bfloat16, device=device)
        pos = make_positions(tokens, "mixed", device)
        fused_ms = bench(lambda: op(pos, e, p, ew, hw, EPS))
        fallback_ms = bench(lambda: fallback(pos, e, p))
        results["perf"][str(tokens)] = {
            "fused_ms": fused_ms,
            "fallback_ms": fallback_ms,
            "speedup": fallback_ms / fused_ms,
        }

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "fused_eh_norm_bf16_20261118.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"results: {path}")


if __name__ == "__main__":
    main()
