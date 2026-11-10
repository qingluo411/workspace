#!/usr/bin/env python3
"""Build a runnable Kimi-K3 layer slice from the full INT4 checkpoint.

The default slice keeps layers 0..23.  This is exactly one quarter of the
93-layer model: it preserves the 1 dense + 23 MoE layer shape, the 3:1 KDA/MLA
ratio, and the first two AttnRes block boundaries.  At TP8 its per-rank weight
footprint is approximately the same as the full model at TP32.

Only real checkpoint tensors are copied.  No random weights are generated.
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


LAYER_RE = re.compile(r"language_model\.model\.layers\.(\d+)\.")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_header(path):
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        return json.loads(handle.read(header_size))


def tensor_nbytes(header, name):
    begin, end = header[name]["data_offsets"]
    return end - begin


def select_names(weight_map, num_layers):
    selected = []
    layer_names = defaultdict(list)
    global_names = []
    max_layer = -1
    for name in weight_map:
        match = LAYER_RE.search(name)
        if match is None:
            global_names.append(name)
            selected.append(name)
            continue
        layer_index = int(match.group(1))
        max_layer = max(max_layer, layer_index)
        if layer_index < num_layers:
            layer_names[layer_index].append(name)
            selected.append(name)
    return selected, global_names, layer_names, max_layer


def filter_ignore(ignore, num_layers):
    filtered = []
    for entry in ignore:
        match = re.search(r"language_model\.model\.layers\.(\d+)\.", entry)
        if match is not None and int(match.group(1)) >= num_layers:
            continue
        filtered.append(entry)
    return filtered


def make_config(source_config, num_layers, output_root):
    config = json.loads(json.dumps(source_config))
    text_config = config["text_config"]
    original_linear = text_config["linear_attn_config"]
    original_kda = set(original_linear["kda_layers"])
    kda_layers = [index + 1 for index in range(num_layers) if index + 1 in original_kda]
    full_attn_layers = [
        index + 1 for index in range(num_layers) if index + 1 not in original_kda
    ]
    text_config["num_hidden_layers"] = num_layers
    text_config["first_k_dense_replace"] = min(
        int(text_config.get("first_k_dense_replace", 1)), num_layers
    )
    text_config["linear_attn_config"] = dict(original_linear)
    text_config["linear_attn_config"]["kda_layers"] = kda_layers
    text_config["linear_attn_config"]["full_attn_layers"] = full_attn_layers
    text_config["num_nextn_predict_layers"] = 0
    text_config["_name_or_path"] = str(output_root)
    ignore = filter_ignore(
        config["quantization_config"].get("ignore", []), num_layers
    )
    config["quantization_config"]["ignore"] = ignore
    text_config["quantization_config"]["ignore"] = json.loads(json.dumps(ignore))
    return config, kda_layers, full_attn_layers, len(ignore)


def copy_non_weight_files(source_root, output_root):
    skipped_suffixes = (".safetensors",)
    rewritten_files = {"config.json", "README.md", "model.safetensors.index.json"}
    for source in source_root.iterdir():
        if not source.is_file():
            continue
        if source.name in rewritten_files:
            continue
        if source.name.endswith(skipped_suffixes):
            continue
        target = output_root / source.name
        shutil.copy2(source, target)


def load_tensor_group(source_root, names, weight_map):
    by_file = defaultdict(list)
    for name in names:
        by_file[weight_map[name]].append(name)
    tensors = {}
    for filename, file_names in by_file.items():
        with safe_open(
            str(source_root / filename), framework="pt", device="cpu"
        ) as handle:
            for name in file_names:
                tensors[name] = handle.get_tensor(name)
    return tensors


def write_shard(source_root, output_root, filename, names, weight_map):
    tensors = load_tensor_group(source_root, names, weight_map)
    output_path = output_root / filename
    save_file(tensors, str(output_path), metadata={"format": "pt"})
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/data/models/Kimi-K3-INT4"))
    parser.add_argument(
        "--output", type=Path, default=Path("/data/models/Kimi-K3-INT4-Slice8-TP8")
    )
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.num_layers <= 0 or args.num_layers > 93:
        raise ValueError("num-layers must be in [1, 93]")
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    source_index = read_json(source_root / "model.safetensors.index.json")
    source_config = read_json(source_root / "config.json")
    weight_map = source_index["weight_map"]

    selected, global_names, layer_names, max_layer = select_names(
        weight_map, args.num_layers
    )
    if max_layer < args.num_layers - 1:
        raise ValueError("checkpoint has fewer layers than requested")
    if any(name not in weight_map for name in selected):
        raise ValueError("selected tensor is absent from the source index")

    headers = {}
    for filename in sorted(set(weight_map.values())):
        headers[filename] = read_header(source_root / filename)
    total_bytes = sum(tensor_nbytes(headers[weight_map[name]], name) for name in selected)
    global_bytes = sum(
        tensor_nbytes(headers[weight_map[name]], name) for name in global_names
    )
    layer_bytes = {
        index: sum(
            tensor_nbytes(headers[weight_map[name]], name)
            for name in layer_names[index]
        )
        for index in range(args.num_layers)
    }
    physical_parameters = sum(
        math.prod(headers[weight_map[name]][name]["shape"]) for name in selected
    )
    config, kda_layers, full_attn_layers, ignore_count = make_config(
        source_config, args.num_layers, output_root
    )

    print(f"source={source_root}")
    print(f"output={output_root}")
    print(f"layers={args.num_layers} tensors={len(selected)}")
    print(f"kda_layers={kda_layers}")
    print(f"full_attn_layers={full_attn_layers}")
    print(f"ignore_entries={ignore_count}")
    print(f"global_bytes={global_bytes:,} layer_bytes={sum(layer_bytes.values()):,}")
    print(f"total_bytes={total_bytes:,} ({total_bytes / 2**30:.2f} GiB)")
    print(f"tp8_weight_per_rank≈{total_bytes / 8 / 2**30:.2f} GiB")
    print(f"physical_parameters={physical_parameters:,}")
    for index in range(args.num_layers):
        print(f"layer {index:02d}: {layer_bytes[index] / 2**30:.2f} GiB")
    if args.dry_run:
        print("dry_run=1")
        return

    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=output_root.name + ".partial-", dir=output_root.parent)
    )
    try:
        copy_non_weight_files(source_root, temp_root)
        write_json(temp_root / "config.json", config)
        (temp_root / "README.md").write_text(
            "# Kimi-K3-INT4 Slice8 (TP8 local)\n\n"
            f"Source: `{source_root}`\n\n"
            f"Layers: `0..{args.num_layers - 1}` ({args.num_layers} total)\n\n"
            f"KDA layers (1-based): `{kda_layers}`\n\n"
            f"Full-attention layers (1-based): `{full_attn_layers}`\n\n"
            f"Data bytes: `{total_bytes}`; TP8 per-rank weight budget: "
            f"`{total_bytes / 8 / 2**30:.2f} GiB`.\n\n"
            "This is a real-weight layer slice, not a synthetic checkpoint. "
            "Use it only for DLC adaptation and deployment smoke tests.\n\n"
            "## Coverage gaps\n\n"
            "- Only layers 0..23 are present; layers 24..92 are not executed.\n"
            "- Only 2 of the original 8 AttnRes blocks are present.\n"
            "- This is single-node TP8, not a cross-node TP32 deployment.\n"
            "- Runtime modes such as prefill/decode, prefix cache, graph, PP/EP/DCP, "
            "and multimodal vision must be exercised separately.\n"
            "- This is a prefix slice, not a distilled model; output semantics do not "
            "represent full Kimi-K3.\n"
        )

        shard_files = []
        shard_start = time.time()
        global_filename = "model-00001-of-00025.safetensors"
        write_shard(source_root, temp_root, global_filename, global_names, weight_map)
        shard_files.append(global_filename)
        print(
            f"wrote {global_filename} global {global_bytes / 2**30:.2f} GiB "
            f"in {time.time() - shard_start:.1f}s",
            flush=True,
        )

        for index in range(args.num_layers):
            shard_number = index + 2
            filename = f"model-{shard_number:05d}-of-00025.safetensors"
            write_shard(source_root, temp_root, filename, layer_names[index], weight_map)
            shard_files.append(filename)
            print(
                f"wrote {filename} layer {index:02d} "
                f"{layer_bytes[index] / 2**30:.2f} GiB in "
                f"{time.time() - shard_start:.1f}s",
                flush=True,
            )

        output_weight_map = {}
        for filename, names in zip(shard_files, [global_names] + [
            layer_names[index] for index in range(args.num_layers)
        ]):
            for name in names:
                output_weight_map[name] = filename
        if set(output_weight_map) != set(selected):
            raise RuntimeError("output weight map does not match selected tensors")

        write_json(
            temp_root / "model.safetensors.index.json",
            {
                "metadata": {
                    "total_size": total_bytes,
                    "physical_parameters": physical_parameters,
                    "slice_of": str(source_root),
                    "slice_layers": args.num_layers,
                    "note": "TP8 local slice with the per-rank weight budget of full TP32",
                },
                "weight_map": output_weight_map,
            },
        )
        write_json(
            temp_root / "slice_manifest.json",
            {
                "source": str(source_root),
                "num_layers": args.num_layers,
                "selected_layers": list(range(args.num_layers)),
                "kda_layers": kda_layers,
                "full_attn_layers": full_attn_layers,
                "global_tensors": len(global_names),
                "layer_tensors": {
                    str(index): len(layer_names[index]) for index in range(args.num_layers)
                },
                "total_bytes": total_bytes,
                "tp8_weight_bytes_per_rank": total_bytes // 8,
                "shards": shard_files,
            },
        )
        output_root_tmp = Path(str(output_root) + f".tmp-{os.getpid()}")
        os.replace(temp_root, output_root_tmp)
        os.replace(output_root_tmp, output_root)
        print(f"done output={output_root}", flush=True)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
