"""Probe whether DLC all-reduce is bit-stable across token-batch shapes."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, world_size: int, port: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if backend == "dlccl":
        torch.dlc.set_device(rank)
        device = f"dlc:{rank}"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32
    dist.init_process_group(
        backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    torch.manual_seed(100 + rank)
    first = (torch.randn((1, 4096), dtype=torch.float32, device=device) * 0.1).to(dtype)
    second = (torch.randn((1, 4096), dtype=torch.float32, device=device) * 0.1).to(dtype)
    one = first.clone()
    two = torch.cat([first, second], dim=0)
    dist.all_reduce(one, op=dist.ReduceOp.SUM)
    dist.all_reduce(two, op=dist.ReduceOp.SUM)
    torch.dlc.synchronize()

    if rank == 0:
        row_one = one[0].float()
        row_two = two[0].float()
        delta = (row_one - row_two).abs()
        print(f"backend={backend} row0_bit_equal={torch.equal(one[0], two[0])}")
        print(f"max_abs={delta.max().item():.9g}")
        print(f"mean_abs={delta.mean().item():.9g}")
        print(f"nonzero={torch.count_nonzero(delta).item()}")
        if backend == "gloo":
            one_bf16 = one.to(torch.bfloat16)
            two_bf16 = two.to(torch.bfloat16)
            bf16_delta = (one_bf16[0].float() - two_bf16[0].float()).abs()
            print(f"row0_bf16_bit_equal={torch.equal(one_bf16[0], two_bf16[0])}")
            print(f"bf16_max_abs={bf16_delta.max().item():.9g}")
            print(f"bf16_nonzero={torch.count_nonzero(bf16_delta).item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="use Gloo CPU all-reduce")
    args = parser.parse_args()
    backend = "gloo" if args.cpu else "dlccl"
    port = 29673 if args.cpu else 29672
    mp.spawn(_worker, args=(8, port, backend), nprocs=8, join=True)
