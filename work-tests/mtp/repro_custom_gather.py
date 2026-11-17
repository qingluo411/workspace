"""Minimal reproduction for the DLC custom_gather_tensor failure.

The graph-capture path in GDNAttentionMetadataBuilder.build() indexes the KDA
block table with a boolean request mask and a two-column slice.  On DLC this
lowers to custom_gather_tensor with the observed input/index shapes.  Run this
after stopping and resetting the model service:

    python work-tests/mtp/repro_custom_gather.py --device dlc:0
"""

from __future__ import annotations

import argparse
import traceback

import torch


def _new_block_table(device: torch.device) -> torch.Tensor:
    # 66560 / 256 = 260 blocks per request, plus one speculative block.
    # Graph-capture padding uses NULL_BLOCK_ID=0.
    return torch.zeros((1, 261), dtype=torch.int32, device=device)


def _run_case(name: str, fn, repeat: int, graph_capture: bool) -> bool:
    try:
        for iteration in range(repeat):
            if graph_capture:
                fn()
                torch.dlc.synchronize()
                graph = torch.dlc.DLCGraph()
                with torch.dlc.graph(graph):
                    output = fn()
                torch.dlc.synchronize()
                graph.replay()
                torch.dlc.synchronize()
                output = output.clone()
            else:
                output = fn()
                torch.dlc.synchronize()
            if iteration == 0:
                print(
                    f"PASS {name}: shape={tuple(output.shape)} dtype={output.dtype}"
                )
        print(f"DONE {name}: iterations={repeat} graph_capture={graph_capture}")
        return True
    except Exception:
        print(f"FAIL {name}")
        traceback.print_exc()
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="dlc:0")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--graph-capture", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)

    with torch.dlc.device(device):
        block_table = _new_block_table(device)
        row_mask_cpu = torch.tensor([True], dtype=torch.bool)
        row_index = torch.zeros((1, 2), dtype=torch.int64, device=device)
        helper_index = torch.tensor([[0, 1]], dtype=torch.int64, device=device)

        cases = [
            (
                "boolean_mask_slice",
                lambda: block_table[row_mask_cpu, :2],
            ),
            (
                "helper_then_boolean_mask",
                lambda: torch.gather(block_table, 1, helper_index)[
                    row_mask_cpu, :2
                ],
            ),
            (
                "mamba_helper_gather",
                lambda: torch.gather(block_table, 1, helper_index),
            ),
            (
                "gather_dim0_rows",
                lambda: torch.gather(block_table, 0, row_index),
            ),
            (
                "gather_dim1_columns",
                lambda: torch.gather(block_table, 1, row_index),
            ),
        ]

        results = [
            _run_case(name, fn, args.repeat, args.graph_capture)
            for name, fn in cases
        ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
