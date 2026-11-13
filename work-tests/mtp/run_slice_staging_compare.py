#!/usr/bin/env python3
"""Compare correct g_a materialization with the implicit slice-staging path."""

from __future__ import annotations

import argparse
import os

import torch

PROMPT_COLS = 3336
HIDDEN_COLS = 4096
HEAD_DIM = 128
OUT_COLS = 1024
G_A_START = 3 * 1024 + 8 + 128
SPLIT_SIZES = [3 * 1024, 8, HEAD_DIM, HEAD_DIM]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("reference", "staging"), required=True)
    parser.add_argument("--device", default="dlc:0")
    parser.add_argument("--result-dir", default="results/slice_staging_compare")
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--staging-mode",
        choices=(
            "view",
            "contiguous",
            "contiguous_sync",
            "sync_before_contiguous",
            "roundtrip",
            "roundtrip_contiguous",
            "clone",
            "index_select",
            "zero_out",
        ),
        default="view",
    )
    return parser.parse_args()


def make_inputs(tokens: int, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "hidden": torch.randn(
            (tokens, HIDDEN_COLS), dtype=torch.bfloat16, device="cpu"
        ),
        "weight_in": torch.randn(
            (PROMPT_COLS, HIDDEN_COLS), dtype=torch.bfloat16, device="cpu"
        ),
        "weight_b": torch.randn(
            (OUT_COLS, HEAD_DIM), dtype=torch.bfloat16, device="cpu"
        ),
    }


def run_in_proj(
    hidden: torch.Tensor,
    weight_in_col: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    projected = torch.empty(
        (hidden.size(0), PROMPT_COLS), dtype=torch.bfloat16, device=device
    )
    torch.dlc_matmul_t_col_major_(projected, hidden, weight_in_col)
    return projected


def run_g_b(
    g_a: torch.Tensor,
    weight_b_col: torch.Tensor,
    device: torch.device,
    zero_out: bool = False,
) -> torch.Tensor:
    out = torch.empty((g_a.size(0), OUT_COLS), dtype=torch.bfloat16, device=device)
    if zero_out:
        out.zero_()
    torch.dlc_matmul_t_col_major_(out, g_a, weight_b_col)
    return out


def save_reference(args: argparse.Namespace, result_dir: str) -> None:
    inputs = make_inputs(args.tokens, args.seed)
    device = torch.device(args.device)
    torch.dlc.set_device(device.index or 0)
    hidden = inputs["hidden"].to(device)
    weight_in = inputs["weight_in"].to(device)
    weight_in_col = torch.dlc_to_col_major_awq(weight_in, weight_in.size(0)).t()
    weight_b_col = torch.dlc_to_col_major_awq(
        inputs["weight_b"].to(device), OUT_COLS
    ).t()
    projected = run_in_proj(hidden, weight_in_col, device)

    index = torch.arange(
        G_A_START, G_A_START + HEAD_DIM, dtype=torch.int64, device=device
    )
    g_a = projected.index_select(1, index)
    out = run_g_b(g_a, weight_b_col, device)
    torch.dlc.synchronize()

    torch.save(inputs, os.path.join(result_dir, "inputs.pt"))
    torch.save(
        {
            "projected": projected.cpu(),
            "g_a": g_a.cpu(),
            "out": out.cpu(),
        },
        os.path.join(result_dir, "reference.pt"),
    )
    print(
        "reference saved: "
        f"projected={tuple(projected.shape)} g_a={tuple(g_a.shape)} "
        f"out={tuple(out.shape)}"
    )


def run_staging(args: argparse.Namespace, result_dir: str) -> int:
    device = torch.device(args.device)
    torch.dlc.set_device(device.index or 0)
    inputs = torch.load(
        os.path.join(result_dir, "inputs.pt"),
        map_location="cpu",
        weights_only=False,
    )
    reference = torch.load(
        os.path.join(result_dir, "reference.pt"),
        map_location="cpu",
        weights_only=False,
    )
    hidden = inputs["hidden"].to(device)
    weight_in = inputs["weight_in"].to(device)
    weight_in_col = torch.dlc_to_col_major_awq(weight_in, weight_in.size(0)).t()
    weight_b_col = torch.dlc_to_col_major_awq(
        inputs["weight_b"].to(device), OUT_COLS
    ).t()
    projected_ref = reference["projected"].to(device)
    g_a_ref = reference["g_a"].to(device)
    out_ref = reference["out"].to(device)

    projected_runs: list[torch.Tensor] = []
    g_a_runs: list[torch.Tensor] = []
    out_runs: list[torch.Tensor] = []
    for _ in range(args.repeat):
        projected = run_in_proj(hidden, weight_in_col, device)
        if args.staging_mode in ("roundtrip", "roundtrip_contiguous"):
            projected = projected.cpu().to(device)
        elif args.staging_mode == "clone":
            projected = projected.clone()
        # This is the non-contiguous view that used to enter the implicit
        # custom_slice_16bit staging path inside the next matmul launch.
        g_a_view = projected.split(SPLIT_SIZES, dim=-1)[-1]
        if args.staging_mode in ("contiguous", "roundtrip_contiguous"):
            g_a_input = g_a_view.contiguous()
        elif args.staging_mode == "contiguous_sync":
            g_a_input = g_a_view.contiguous()
            torch.dlc.synchronize()
        elif args.staging_mode == "sync_before_contiguous":
            torch.dlc.synchronize()
            g_a_input = g_a_view.contiguous()
        elif args.staging_mode == "index_select":
            index = torch.arange(
                G_A_START,
                G_A_START + HEAD_DIM,
                dtype=torch.int64,
                device=device,
            )
            g_a_input = projected.index_select(1, index)
        else:
            g_a_input = g_a_view
        out = run_g_b(
            g_a_input,
            weight_b_col,
            device,
            zero_out=args.staging_mode == "zero_out",
        )
        projected_runs.append(projected)
        g_a_runs.append(g_a_input)
        out_runs.append(out)
    torch.dlc.synchronize()

    failed_iters = []
    failed_elements = 0
    projected_failures = 0
    g_a_failures = 0
    first_failure = None
    for index, (projected, g_a_input, out) in enumerate(
        zip(projected_runs, g_a_runs, out_runs)
    ):
        projected_diff = int((projected != projected_ref).sum().item())
        g_a_diff = int((g_a_input != g_a_ref).sum().item())
        out_diff = int((out != out_ref).sum().item())
        projected_failures += int(projected_diff != 0)
        g_a_failures += int(g_a_diff != 0)
        if out_diff:
            failed_iters.append((index, projected_diff, g_a_diff, out_diff))
            failed_elements += out_diff
            if first_failure is None:
                g_a_view_cpu = projected.cpu().split(SPLIT_SIZES, dim=-1)[-1]
                first_failure = {
                    "iteration": index,
                    "projected": projected.cpu(),
                    "g_a": g_a_view_cpu,
                    "g_a_input": g_a_input.cpu(),
                    "out": out.cpu(),
                    "out_ref": out_ref.cpu(),
                }

    print(
        "staging compare: "
        f"iterations={args.repeat} "
        f"mode={args.staging_mode} "
        f"projected_failures={projected_failures} "
        f"g_a_failures={g_a_failures} "
        f"failed_iters={len(failed_iters)} "
        f"failed_elements={failed_elements}"
    )
    if failed_iters:
        print(f"first failures: {failed_iters[:5]}")
        first = failed_iters[0][0]
        row_diff = [
            (out_runs[first][row] != out_ref[row]).sum().item()
            for row in range(out_runs[first].size(0))
        ]
        print(f"first failure row diff: {row_diff}")
        torch.save(first_failure, os.path.join(result_dir, "first_failure.pt"))
    torch.save(
        {
            "g_a": g_a_runs[-1].cpu(),
            "out": out_runs[-1].cpu(),
        },
        os.path.join(result_dir, "staging.pt"),
    )
    return 0 if not failed_iters else 1


def main() -> int:
    args = parse_args()
    result_dir = os.path.abspath(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)
    if args.phase == "reference":
        save_reference(args, result_dir)
        return 0
    return run_staging(args, result_dir)


if __name__ == "__main__":
    raise SystemExit(main())
