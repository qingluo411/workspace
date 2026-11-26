#!/usr/bin/env python3
"""Compare two full model-dump iterations and report the first differing file."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


def iter_tensor_leaves(value, path=""):
    if isinstance(value, torch.Tensor):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield from iter_tensor_leaves(item, child)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            child = f"{path}.{index}" if path else str(index)
            yield from iter_tensor_leaves(item, child)


def tensor_hash(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_file(path: Path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    return {
        leaf_path: {
            "tensor": tensor,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "hash": tensor_hash(tensor),
        }
        for leaf_path, tensor in iter_tensor_leaves(value)
    }


def tensor_diff(left: torch.Tensor, right: torch.Tensor) -> dict:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "shape_equal": list(left.shape) == list(right.shape),
            "dtype_equal": left.dtype == right.dtype,
        }
    left_f = left.float()
    right_f = right.float()
    delta = (left_f - right_f).abs()
    nonzero = int((delta != 0).sum().item())
    flat = delta.reshape(-1)
    top = torch.topk(flat, k=min(8, flat.numel()))
    return {
        "mismatch_elements": nonzero,
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
        "top_abs_diff": top.values.tolist(),
        "top_flat_indices": top.indices.tolist(),
    }


def compare_iterations(left_dir: Path, right_dir: Path, output: Path | None):
    left_files = {
        path.name: path
        for path in left_dir.iterdir()
        if path.suffix == ".pt"
    }
    right_files = {
        path.name: path
        for path in right_dir.iterdir()
        if path.suffix == ".pt"
    }
    differences = []
    for name in sorted(set(left_files) | set(right_files)):
        if name not in left_files or name not in right_files:
            differences.append(
                {
                    "file": name,
                    "reason": "missing",
                    "left": name in left_files,
                    "right": name in right_files,
                }
            )
            continue
        left = load_file(left_files[name])
        right = load_file(right_files[name])
        for leaf_path in sorted(set(left) | set(right)):
            left_leaf = left.get(leaf_path)
            right_leaf = right.get(leaf_path)
            if left_leaf is None or right_leaf is None:
                differences.append(
                    {
                        "file": name,
                        "path": leaf_path,
                        "reason": "missing_leaf",
                    }
                )
                continue
            if (
                left_leaf["hash"] == right_leaf["hash"]
                and left_leaf["shape"] == right_leaf["shape"]
                and left_leaf["dtype"] == right_leaf["dtype"]
            ):
                continue
            differences.append(
                {
                    "file": name,
                    "path": leaf_path,
                    "reason": "value_mismatch",
                    "dtype": left_leaf["dtype"],
                    "shape": left_leaf["shape"],
                    "left_hash": left_leaf["hash"],
                    "right_hash": right_leaf["hash"],
                    "diff": tensor_diff(
                        left_leaf["tensor"], right_leaf["tensor"]
                    ),
                }
            )
        if differences:
            break

    summary = {
        "left_dir": str(left_dir),
        "right_dir": str(right_dir),
        "left_files": len(left_files),
        "right_files": len(right_files),
        "first_difference": differences[0] if differences else None,
        "same_through_first_difference": not differences,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not differences else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left_dir", type=Path)
    parser.add_argument("right_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    return compare_iterations(args.left_dir, args.right_dir, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
