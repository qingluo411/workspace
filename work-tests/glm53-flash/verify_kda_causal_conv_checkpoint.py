"""Validate KDA causal-conv checkpoint export against prefix-only state."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

RESULTS = Path("/root/workspace/work-tests/glm53-flash/results")


def make_inputs(lengths, dim, state_len, slots, has_init, seed, device):
    torch.manual_seed(seed)
    batch = len(lengths)
    total = sum(lengths)
    qkv = torch.randn(total, dim, dtype=torch.bfloat16, device=device)
    weight = torch.randn(4, dim, dtype=torch.float32, device=device)
    state = torch.randn(slots, state_len, dim, dtype=torch.bfloat16, device=device)
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    qsl = torch.tensor(starts, dtype=torch.int32, device=device)
    indices = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    hinit = torch.tensor(has_init, dtype=torch.int32, device=device)
    return qkv, weight, state, qsl, indices, hinit


def run_operator(qkv, weight, state, qsl, indices, hinit, offsets=None, ckpt=None):
    kwargs = {}
    if offsets is not None:
        kwargs = {"checkpoint_offsets": offsets, "checkpoint_state": ckpt}
    return torch.dlc_kda_causal_conv_prefill(
        qkv,
        weight,
        state,
        qsl,
        indices,
        hinit,
        activation="silu",
        null_state_id=0,
        **kwargs,
    )


def check_prefix_state(qkv, weight, state, qsl, indices, hinit, offsets):
    batch = qsl.numel() - 1
    ckpt = torch.full((batch, state.shape[1], state.shape[2]), 3.0,
                      dtype=torch.bfloat16, device=qkv.device)
    state_before = state.clone()
    out = run_operator(qkv, weight, state, qsl, indices, hinit, offsets, ckpt)
    torch.dlc.synchronize()
    rows = []
    for b in range(batch):
        offset = int(offsets[b])
        slot = int(indices[b])
        if offset == 0 or slot == 0:
            rows.append({"row": b, "offset": offset, "skipped": True,
                         "unchanged": bool(torch.equal(ckpt[b], torch.full_like(ckpt[b], 3.0)))})
            continue
        start = int(qsl[b])
        prefix_state = state_before.clone()
        prefix_out = run_operator(
            qkv[start:start + offset].contiguous(),
            weight,
            prefix_state,
            torch.tensor([0, offset], dtype=torch.int32, device=qkv.device),
            indices[b:b + 1],
            hinit[b:b + 1],
        )
        torch.dlc.synchronize()
        expected = prefix_state[slot]
        diff = (ckpt[b].float() - expected.float()).abs().max().item()
        rows.append({"row": b, "offset": offset, "slot": slot,
                     "max_abs_diff": diff, "bit_equal": bool(torch.equal(ckpt[b], expected))})
    return out, rows


def bench(fn, warmup=5, repeat=20):
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
    results = {"device": str(device), "state_checks": [], "perf": {}, "errors": {}}

    cases = [
        ([4096], 3072, 6, 2, [True], [3584], 101),
        ([65536], 256, 6, 2, [True], [65280], 102),
        ([65536], 3072, 6, 2, [True], [65280], 105),
        ([4096, 7168], 512, 6, 3, [True, False], [3584, 6912], 103),
        ([128, 128], 256, 80, 3, [True, False], [64, 64], 104),
    ]
    for lengths, dim, state_len, slots, hinit, offsets, seed in cases:
        qkv, weight, state, qsl, indices, hinit_t = make_inputs(
            lengths, dim, state_len, slots, hinit, seed, device
        )
        offs = torch.tensor(offsets, dtype=torch.int32, device=device)
        _, rows = check_prefix_state(qkv, weight, state, qsl, indices, hinit_t, offs)
        results["state_checks"].append({"lengths": lengths, "dim": dim,
                                        "state_len": state_len, "offsets": offsets,
                                        "rows": rows})

    # No-checkpoint and all-zero offset paths must be identical.
    qkv, weight, state, qsl, indices, hinit_t = make_inputs(
        [4096], 3072, 6, 2, [True], 7, device
    )
    s0 = state.clone()
    s1 = state.clone()
    out0 = run_operator(qkv, weight, s0, qsl, indices, hinit_t)
    ckpt = torch.full((1, 6, 3072), 3.0, dtype=torch.bfloat16, device=device)
    out1 = run_operator(qkv, weight, s1, qsl, indices, hinit_t,
                        torch.zeros(1, dtype=torch.int32, device=device), ckpt)
    torch.dlc.synchronize()
    results["zero_offset_equal"] = {
        "out": bool(torch.equal(out0, out1)),
        "state": bool(torch.equal(s0, s1)),
        "checkpoint_untouched": bool(torch.equal(ckpt, torch.full_like(ckpt, 3.0))),
    }

    # Performance: same launch count; checkpoint adds in-kernel snapshot/store.
    for tokens, dim, state_len, offset in (
        (4096, 3072, 6, 3584),
        (65536, 256, 6, 65280),
        (65536, 3072, 6, 65280),
    ):
        qkv, weight, state, qsl, indices, hinit_t = make_inputs(
            [tokens], dim, state_len, 2, [True], 200 + tokens, device
        )
        offs = torch.tensor([offset], dtype=torch.int32, device=device)
        ckpt = torch.full((1, state_len, dim), 3.0, dtype=torch.bfloat16, device=device)
        state0 = state.clone()
        state1 = state.clone()
        base = bench(lambda: run_operator(qkv, weight, state0, qsl, indices, hinit_t))
        fused = bench(lambda: run_operator(qkv, weight, state1, qsl, indices, hinit_t, offs, ckpt))
        results["perf"][f"{tokens}_d{dim}"] = {
            "dim": dim, "offset": offset,
            "base_ms": base, "checkpoint_ms": fused,
            "overhead_pct": (fused / base - 1.0) * 100.0,
            "extra_launch": 0,
        }

    # Host-side API validation errors (safe, before kernel launch).
    qkv, weight, state, qsl, indices, hinit_t = make_inputs(
        [128], 256, 5, 2, [True], 300, device
    )
    offs = torch.tensor([64], dtype=torch.int32, device=device)
    good_ckpt = torch.zeros(1, 5, 256, dtype=torch.bfloat16, device=device)
    noncontig_base = torch.zeros(1, 5, 512, dtype=torch.bfloat16, device=device)
    noncontig_ckpt = noncontig_base[:, :, ::2]
    error_cases = {
        "offsets_without_state": lambda: run_operator(qkv, weight, state, qsl, indices, hinit_t, offs, None),
        "state_without_offsets": lambda: torch.dlc_kda_causal_conv_prefill(
            qkv, weight, state, qsl, indices, hinit_t,
            activation="silu", null_state_id=0,
            checkpoint_state=good_ckpt,
        ),
        "wrong_state_shape": lambda: run_operator(qkv, weight, state, qsl, indices, hinit_t, offs, torch.zeros(1, 5, 128, dtype=torch.bfloat16, device=device)),
        "wrong_state_dtype": lambda: run_operator(qkv, weight, state, qsl, indices, hinit_t, offs, torch.zeros(1, 5, 256, dtype=torch.float32, device=device)),
        "noncontiguous_state": lambda: run_operator(qkv, weight, state, qsl, indices, hinit_t, offs, noncontig_ckpt),
        "wrong_offsets_dtype": lambda: run_operator(qkv, weight, state, qsl, indices, hinit_t, offs.long(), good_ckpt),
    }
    for name, fn in error_cases.items():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            results["errors"][name] = type(exc).__name__
        else:
            raise AssertionError(f"expected error for {name}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "kda_causal_conv_checkpoint_20261118.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"results: {path}")


if __name__ == "__main__":
    main()
