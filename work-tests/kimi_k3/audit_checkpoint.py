#!/usr/bin/env python3
"""Audit the Kimi-K3-INT4 checkpoint without loading tensor data."""

import argparse
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def read_header(path):
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return header_size, header


def category(name):
    if name.startswith("vision_tower."):
        return "vision_tower"
    if name.startswith("mm_projector."):
        return "mm_projector"
    if "embed_tokens" in name:
        return "embedding"
    if name.startswith("language_model.lm_head"):
        return "lm_head"
    if ".block_sparse_moe.experts." in name:
        if name.endswith(".weight_packed"):
            return "routed_experts_packed"
        if name.endswith(".weight_scale"):
            return "routed_experts_scale"
        if name.endswith(".weight_shape"):
            return "routed_experts_shape"
        return "routed_experts_other"
    if ".block_sparse_moe.shared_experts." in name:
        return "shared_experts"
    if ".block_sparse_moe.gate." in name:
        return "moe_gate"
    if "routed_expert_down_proj" in name or "routed_expert_up_proj" in name:
        return "latent_moe_transform"
    if "routed_expert_norm" in name:
        return "latent_moe_norm"
    if "attn_res" in name:
        return "attn_res"
    if ".self_attn." in name or ".self_attention" in name:
        return "attention"
    if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
        return "norm"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data/models/Kimi-K3-INT4"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = json.loads((args.model / "config.json").read_text())
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    expected = set(weight_map)

    files = sorted(set(weight_map.values()))
    category_bytes = Counter()
    category_tensors = Counter()
    dtype_bytes = Counter()
    dtype_tensors = Counter()
    seen = set()
    shard_errors = []
    shard_bytes = {}

    for filename in files:
        path = args.model / filename
        if not path.is_file():
            shard_errors.append(f"missing shard: {filename}")
            continue
        header_size, header = read_header(path)
        data_end = 8 + header_size
        file_size = path.stat().st_size
        shard_total = 0
        for tensor_name, info in header.items():
            if tensor_name == "__metadata__":
                continue
            if tensor_name not in expected:
                shard_errors.append(f"unexpected tensor: {tensor_name} in {filename}")
                continue
            if tensor_name in seen:
                shard_errors.append(f"duplicate tensor: {tensor_name}")
                continue
            begin, end = info["data_offsets"]
            if begin < 0 or end < begin:
                shard_errors.append(f"invalid offsets: {tensor_name}")
                continue
            nbytes = end - begin
            dtype = info["dtype"]
            expected_nbytes = DTYPE_BYTES.get(dtype)
            if expected_nbytes is None:
                shard_errors.append(f"unknown dtype {dtype}: {tensor_name}")
            else:
                elements = 1
                for dim in info["shape"]:
                    elements *= dim
                if elements * expected_nbytes != nbytes:
                    shard_errors.append(
                        f"size mismatch {tensor_name}: dtype={dtype} "
                        f"shape={info['shape']} bytes={nbytes}"
                    )
            data_end = max(data_end, 8 + header_size + end)
            seen.add(tensor_name)
            cat = category(tensor_name)
            category_bytes[cat] += nbytes
            category_tensors[cat] += 1
            dtype_bytes[dtype] += nbytes
            dtype_tensors[dtype] += 1
            shard_total += nbytes
        shard_bytes[filename] = shard_total
        if data_end > file_size:
            shard_errors.append(
                f"truncated shard {filename}: needs {data_end}, has {file_size}"
            )

    missing = sorted(expected - seen)
    if missing:
        shard_errors.append(f"{len(missing)} indexed tensors are absent from headers")

    quant_triplets = defaultdict(set)
    for name in expected:
        if ".block_sparse_moe.experts." not in name:
            continue
        if name.endswith(".weight_packed"):
            suffix = ".weight_packed"
        elif name.endswith(".weight_scale"):
            suffix = ".weight_scale"
        elif name.endswith(".weight_shape"):
            suffix = ".weight_shape"
        else:
            continue
        quant_triplets[name[: -len(suffix)]].add(suffix)
    incomplete_triplets = {
        base: sorted(suffixes)
        for base, suffixes in quant_triplets.items()
        if len(suffixes) != 3
    }

    text_config = config["text_config"]
    linear_config = text_config["linear_attn_config"]
    layer_count = text_config["num_hidden_layers"]
    kda_layers = set(linear_config["kda_layers"])
    mla_layers = sorted(
        idx + 1 for idx in range(layer_count) if idx + 1 not in kda_layers
    )
    index_metadata = index.get("metadata", {})
    total_parameters = index_metadata.get("total_parameters")
    total_size = index_metadata.get("total_size", sum(category_bytes.values()))
    report = {
        "model": str(args.model),
        "parameters": total_parameters,
        "indexed_size": total_size,
        "tensor_count": len(expected),
        "shard_count": len(files),
        "shards": shard_bytes,
        "category_bytes": dict(category_bytes),
        "category_tensors": dict(category_tensors),
        "dtype_bytes": dict(dtype_bytes),
        "dtype_tensors": dict(dtype_tensors),
        "layers": layer_count,
        "kda_layers": len(kda_layers),
        "mla_layers_1_based": mla_layers,
        "quantized_expert_triplets": len(quant_triplets),
        "incomplete_expert_triplets": incomplete_triplets,
        "errors": shard_errors,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"tensors={len(expected)} shards={len(files)}")
    if total_parameters is not None:
        print(f"parameters={total_parameters:,}")
    print(f"bytes={total_size:,}")
    print(f"kda_layers={len(kda_layers)} mla_layers={len(mla_layers)}")
    print(f"expert_triplets={len(quant_triplets)} incomplete={len(incomplete_triplets)}")
    for name, nbytes in category_bytes.most_common():
        print(f"{name:28s} {category_tensors[name]:8d} {nbytes:16d}")
    if shard_errors:
        print(f"errors={len(shard_errors)}")
        for error in shard_errors[:20]:
            print(f"  {error}")
    else:
        print("errors=0")
    if args.output:
        print(f"report={args.output}")


if __name__ == "__main__":
    main()
