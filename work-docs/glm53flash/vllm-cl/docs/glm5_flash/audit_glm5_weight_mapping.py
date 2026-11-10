#!/usr/bin/env python3
"""Audit GLM-5.3 AWQ checkpoint names and headers without loading tensors.

The text-model mapping executes the current Glm5NextModel.load_weights source
against recording parameters. The independent expected catalog below follows
the constructors in glm5next/nvidia/{model,attention,multimodal}.py and
glm5next/nvidia/linear_attention.py. It deliberately supports the initial
TP=EP=PP=1, no-MTP, BF16/AWQ checkpoint contract, not arbitrary GLM variants.
No vLLM, torch, DLC library or safetensors tensor payload is imported/read.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import struct
from types import SimpleNamespace


def class_node(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == name)


def mapper_dict(path, name, keyword):
    cls = class_node(path, name)
    value = next(n.value for n in cls.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "hf_to_vllm_mapper"
                         for t in n.targets))
    return ast.literal_eval(next(k.value for k in value.keywords if k.arg == keyword))


def make_catalog(text, vision, quant):
    """Independent checkpoint catalog: shape, dtype, target and target shard."""
    expected = {}
    h, i, m = text["hidden_size"], text["intermediate_size"], text["moe_intermediate_size"]
    ne, nh, hc = text["n_routed_experts"], text["num_attention_heads"], text["hc_mult"]
    q, kv = text["q_lora_rank"], text["kv_lora_rank"]
    idx, ih = text["index_head_dim"], text["index_n_heads"]
    la = text["linear_attn_config"]
    ld, lh = la["head_dim"], la["num_heads"]
    packed, group = 32 // quant["bits"], quant["group_size"]

    def add(source, shape, category, *, dtype="BF16", target=None, shard=None,
            expert=None, skip=False):
        assert source not in expected, source
        expected[source] = dict(shape=list(shape), dtype=dtype, category=category,
                                target=target or source, shard=shard, expert=expert,
                                skip=skip)

    def text_add(prefix, tail, shape, category, **kwargs):
        source = "model.language_model." + prefix + tail
        target = kwargs.pop("target", tail)
        add(source, shape, category,
            target="language_model.model." + prefix + target, **kwargs)

    add("lm_head.weight", (text["vocab_size"], h), "lm_head",
        target="language_model.lm_head.weight")
    text_add("", "embed_tokens.weight", (text["vocab_size"], h), "embedding")
    text_add("", "norm.weight", (h,), "final_norm")
    for layer in range(text["num_hidden_layers"] + text["num_nextn_predict_layers"]):
        pre = f"layers.{layer}."
        mtp = layer >= text["num_hidden_layers"]
        cat = "mtp_skipped" if mtp else "layer_norm_mhc"

        def emit(tail, shape, category=cat, **kwargs):
            text_add(pre, tail, shape, "mtp_skipped" if mtp else category,
                     skip=mtp, **kwargs)

        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            emit(name, (h,))
        if mtp:
            for name in ("enorm.weight", "hnorm.weight", "shared_head.norm.weight"):
                emit(name, (h,))
            emit("eh_proj.weight", (h, 2 * h))
        else:
            for kind in ("attn", "ffn"):
                emit(f"hc_{kind}_fn", (hc * (hc + 2), hc * h))
                emit(f"hc_{kind}_base", (hc * (hc + 2),), dtype="F32")
                emit(f"hc_{kind}_scale", (3,), dtype="F32")

        sparse = mtp or text["mlp_layer_types"][layer] == "sparse"
        if not sparse:
            for shard, name in enumerate(("gate_proj", "up_proj")):
                emit(f"mlp.{name}.weight", (i, h), "dense_bf16",
                     target="mlp.gate_up_proj.weight", shard=shard)
            emit("mlp.down_proj.weight", (h, i), "dense_bf16")
        else:
            emit("mlp.gate.weight", (ne, h), "router_bf16_to_fp32")
            emit("mlp.gate.e_score_correction_bias", (ne,), "router_fp32", dtype="F32")
            shared = m * text["n_shared_experts"]
            for shard, name in enumerate(("gate_proj", "up_proj")):
                emit(f"mlp.shared_experts.{name}.weight", (shared, h), "shared_bf16",
                     target="mlp.shared_experts.gate_up_proj.weight", shard=shard)
            emit("mlp.shared_experts.down_proj.weight", (h, shared), "shared_bf16")
            for expert in range(ne):
                for proj, shard in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")):
                    input_size, output_size = (m, h) if proj == "down_proj" else (h, m)
                    target = "w2" if proj == "down_proj" else "w13"
                    params = {"weight": ((output_size, input_size), "BF16")} if mtp else {
                        "qweight": ((input_size, output_size // packed), "I32"),
                        "qzeros": ((input_size // group, output_size // packed), "I32"),
                        "scales": ((input_size // group, output_size), "BF16"),
                    }
                    for param, (shape, dtype) in params.items():
                        emit(f"mlp.experts.{expert}.{proj}.{param}", shape, "routed_awq",
                             dtype=dtype, expert=expert, shard=shard,
                             target=f"mlp.experts.routed_experts.{target}_{param}")

        linear = not mtp and text["layer_types"][layer] == "linear_attention"
        if linear:
            for shard, (proj, rows) in enumerate((("q", lh * ld), ("k", lh * ld),
                                                 ("v", lh * ld), ("b", lh),
                                                 ("f_a", ld), ("g_a", ld))):
                emit(f"self_attn.{proj}_proj.weight", (rows, h), "kda_bf16",
                     target="self_attn.in_proj_qkvbfg_a.weight", shard=shard)
            for proj in ("f_b", "g_b"):
                emit(f"self_attn.{proj}_proj.weight", (lh * ld, ld), "kda_bf16")
            for proj in ("q", "k", "v"):
                emit(f"self_attn.{proj}_conv1d.weight", (lh * ld, 1, la["short_conv_kernel_size"]), "kda_bf16")
            emit("self_attn.A_log", (lh,), "kda_fp32", dtype="F32")
            emit("self_attn.dt_bias", (lh * ld,), "kda_fp32", dtype="F32")
            emit("self_attn.o_norm.weight", (ld,), "kda_bf16")
            emit("self_attn.o_proj.weight", (h, lh * ld), "kda_bf16")
        else:
            specs = {
                "q_a_layernorm.weight": (q,), "kv_a_layernorm.weight": (kv,),
                "q_b_proj.weight": (nh * text["qk_nope_head_dim"], q),
                "kv_b_proj.weight": (nh * (text["qk_nope_head_dim"] + text["v_head_dim"]), kv),
                "o_proj.weight": (h, nh * text["v_head_dim"]),
                "indexer.index_kpool_compress_ape": (text["index_kpool"], idx),
                "indexer.index_kpool_compress_gate": (idx, h),
                "indexer.k_norm.weight": (idx,), "indexer.k_norm.bias": (idx,),
                "indexer.wq_b.weight": (ih * idx, q),
            }
            for tail, shape in specs.items():
                emit("self_attn." + tail, shape, "mla_indexer_bf16")
            for tail, rows, target, shard in (
                ("q_a_proj", q, "fused_qkv_a_proj", 0),
                ("kv_a_proj_with_mqa", kv, "fused_qkv_a_proj", 1),
                ("indexer.wk", idx, "indexer.wk_weights_proj", 0),
                ("indexer.weights_proj", ih, "indexer.wk_weights_proj", 1),
            ):
                emit("self_attn." + tail + ".weight", (rows, h), "mla_indexer_bf16",
                     target="self_attn." + target + ".weight", shard=shard)

    vh, vi, vo = vision["hidden_size"], vision["intermediate_size"], vision["out_hidden_size"]

    def vis(tail, shape, *, target=None, shard=None):
        add("model.visual." + tail, shape, "vision_bf16",
            target="visual." + (target or tail), shard=shard)

    for block in range(vision["depth"]):
        pre = f"blocks.{block}."
        for norm in ("norm1", "norm2"):
            vis(pre + norm + ".weight", (vh,))
        for norm in ("q_norm", "k_norm"):
            vis(pre + "attn." + norm + ".weight", (vh // vision["num_heads"],))
        for name, rows in (("qkv", 3 * vh), ("proj", vh)):
            vis(pre + f"attn.{name}.weight", (rows, vh))
            vis(pre + f"attn.{name}.bias", (rows,))
        vis(pre + "mlp.down_proj.weight", (vh, vi))
        vis(pre + "mlp.down_proj.bias", (vh,))
        for shard, name in enumerate(("gate_proj", "up_proj")):
            for param, shape in (("weight", (vi, vh)), ("bias", (vi,))):
                vis(pre + f"mlp.{name}.{param}", shape,
                    target=pre + f"mlp.gate_up_proj.{param}", shard=shard)
    for tail, shape in {
        "patch_embed.proj.weight": (vh, vision["in_channels"], vision["temporal_patch_size"], vision["patch_size"], vision["patch_size"]),
        "patch_embed.proj.bias": (vh,), "post_layernorm.weight": (vh,),
        "downsample.weight": (vo, vh, vision["spatial_merge_size"], vision["spatial_merge_size"]),
        "downsample.bias": (vo,), "merger.proj.weight": (vo, vo),
        "merger.post_projection_norm.weight": (vo,), "merger.post_projection_norm.bias": (vo,),
        "merger.down_proj.weight": (vo, vision["projection_intermediate_size"]),
    }.items():
        vis(tail, shape)
    for shard, name in enumerate(("gate_proj", "up_proj")):
        vis(f"merger.{name}.weight", (vision["projection_intermediate_size"], vo),
            target="merger.gate_up_proj.weight", shard=shard)
    return expected


def audit(model_dir, vllm_dir):
    config = json.loads((model_dir / "config.json").read_text())
    quant = json.loads((model_dir / "quantize_config.json").read_text())
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    text_config = config["text_config"]
    if text_config["qk_rope_head_dim"] != 0 or quant["bits"] != 4:
        raise ValueError("This audit catalog currently requires NoPE and 4-bit AWQ")
    root = vllm_dir / "vllm"
    model_source = root / "models/glm5next/nvidia/model.py"
    vision_source = root / "models/glm5next/nvidia/multimodal.py"
    outer = mapper_dict(root / "model_executor/models/glm4_1v.py", "Glm4vForConditionalGeneration", "orig_to_new_prefix")
    vision_mapper = mapper_dict(vision_source, "Glm5NextVisionTransformer", "orig_to_new_stacked")
    catalog = make_catalog(text_config, config["vision_config"], quant)
    # This is the exact transformation done by
    # model_loader.utils.configure_quant_config -> AutoAWQConfig.apply_vllm_mapper.
    # Keep both namespaces in the report because reviewing only one is a common
    # source of accidentally quantizing the first three dense layers.
    mapped_skip_modules = []
    for module in quant.get("modules_to_not_convert", []):
        mapped = module
        for old, new in outer.items():
            if mapped.startswith(old):
                mapped = new + mapped[len(old):]
                break
        mapped_skip_modules.append(mapped)
    skip_match_counts = {
        module: sum(1 for key, entry in catalog.items()
                    if entry["target"].startswith(module) or module in entry["target"])
        for module in mapped_skip_modules
    }
    # The outer GLM-4.1V class currently exposes gate_up_proj:['gate_up_proj'].
    # AutoAWQ's fused-layer skip check expands a fused prefix through this
    # mapping. Consequently GLM-5's three dense layers do not match their
    # checkpoint gate_proj/up_proj exclusions until the mapping is corrected.
    current_packed_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"],
                              "gate_up_proj": ["gate_up_proj"]}
    required_packed_mapping = {**current_packed_mapping,
                               "gate_up_proj": ["gate_proj", "up_proj"]}
    dense_skip_analysis = {}
    for fused_name, packed in (("current", current_packed_mapping),
                               ("required", required_packed_mapping)):
        dense_skip_analysis[fused_name] = {
            str(layer): {
                "gate_up_proj_skipped": all(
                    any(mod in f"language_model.model.layers.{layer}.mlp.{shard}"
                        for mod in mapped_skip_modules)
                    for shard in packed["gate_up_proj"]
                )
            }
            for layer in range(3)
        }
    issues = []
    headers = {}
    unavailable_shards = []
    incomplete_payloads = []
    for filename in sorted(set(weight_map.values())):
        path = model_dir / filename
        try:
            with path.open("rb") as handle:
                header_size = struct.unpack("<Q", handle.read(8))[0]
                if header_size > 256 * 1024 * 1024:
                    raise ValueError("implausible safetensors header size")
                metadata = json.loads(handle.read(header_size))
            metadata.pop("__metadata__", None)
            headers[filename] = metadata
            required = 8 + header_size + max((v["data_offsets"][1] for v in metadata.values()), default=0)
            if path.stat().st_size < required:
                incomplete_payloads.append(filename)
        except (OSError, ValueError, struct.error) as exc:
            unavailable_shards.append(dict(shard=filename, reason=str(exc)))
    missing_keys = sorted(set(catalog) - set(weight_map))
    unexpected_keys = sorted(set(weight_map) - set(catalog))
    for filename, header in headers.items():
        for key in header:
            if weight_map.get(key) != filename:
                issues.append(dict(kind="header_index_mismatch", key=key, shard=filename))

    metadata_by_key = {}
    for key, filename in weight_map.items():
        header = headers.get(filename)
        if header is None:
            continue
        if key not in header:
            issues.append(dict(kind="index_key_missing_from_header", key=key, shard=filename))
            continue
        meta = metadata_by_key[key] = header[key]
        if key in catalog:
            exp = catalog[key]
            if meta["shape"] != exp["shape"] or meta["dtype"] != exp["dtype"]:
                issues.append(dict(kind="shape_or_dtype", key=key,
                                   expected={k: exp[k] for k in ("shape", "dtype")},
                                   actual={k: meta[k] for k in ("shape", "dtype")}))

    records = {}
    slots = {}

    def record(source, target, shard=None, expert=None):
        actual = dict(target=target, shard=shard, expert=expert)
        records.setdefault(source, []).append(actual)
        slot = (target, shard, expert)
        if slot in slots:
            issues.append(dict(kind="duplicate_target_slot", keys=[slots[slot], source], target=actual))
        slots[slot] = source
        exp = catalog[source]
        if actual != {k: exp[k] for k in actual}:
            issues.append(dict(kind="loader_mapping", key=source, actual=actual,
                               expected={k: exp[k] for k in actual}))

    class RecordingParam:
        def __init__(self, name):
            self.name = name

        def weight_loader(self, param, value, *args, **kwargs):
            expert = kwargs.get("expert_id")
            shard = kwargs.get("shard_id", args[0] if args and expert is None else None)
            record(value.source, "language_model.model." + self.name, shard, expert)
            return True

    def expert_mapping(*args, **kwargs):
        return [(f"experts.routed_experts.{target}_", f"experts.{expert}.{proj}.", expert, shard)
                for expert in range(kwargs["num_experts"])
                for target, proj, shard in (("w13", "gate_proj", "w1"),
                                             ("w2", "down_proj", "w2"),
                                             ("w13", "up_proj", "w3"))]

    loader_ast = next(n for n in class_node(model_source, "Glm5NextModel").body
                      if isinstance(n, ast.FunctionDef) and n.name == "load_weights")
    loader_ast.decorator_list = []
    namespace = dict(
        fused_moe_make_expert_params_mapping=expert_mapping,
        Glm5NextDecoderLayer=type("Decoder", (), {}),
        Glm5NextMoE=type("MoE", (), {}),
        get_spec_layer_idx_from_weight_name=lambda config, name: int(name.split(".")[1])
        if name.startswith("layers.") and int(name.split(".")[1]) >= config.num_hidden_layers else None,
        _try_load_fp8_indexer_wk=lambda *args: False,
        _try_load_fp8_attn_proj=lambda *args: False,
        is_pp_missing_parameter=lambda *args: False,
        maybe_remap_kv_scale_name=lambda name, params: name,
        default_weight_loader=lambda param, value, **kwargs: param.weight_loader(param, value, **kwargs),
    )
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), loader_ast], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(model_source), "exec"), namespace)
    prefix = "language_model.model."
    params = {entry["target"][len(prefix):]: RecordingParam(entry["target"][len(prefix):])
              for entry in catalog.values() if not entry["skip"] and entry["target"].startswith(prefix)}
    model = SimpleNamespace(config=SimpleNamespace(**text_config, is_moe=True,
                                                  mla_nope=text_config["mla_use_nope"]),
                            layers=[], named_parameters=lambda: params.items())
    main_weights = []
    for source in weight_map:
        if source not in catalog:
            continue
        mapped = source
        for old, new in outer.items():
            if mapped.startswith(old):
                mapped = new + mapped[len(old):]
                break
        if mapped.startswith(prefix):
            exp = catalog[source]
            # Shape/dtype are independently checked against actual headers above;
            # the recording loader never reads or computes on tensor payloads.
            main_weights.append((mapped[len(prefix):], SimpleNamespace(source=source,
                                                                       shape=exp["shape"], dtype=exp["dtype"])))
        elif mapped.startswith("visual."):
            shard = None
            for old, (new, sid) in vision_mapper.items():
                if old in mapped:
                    mapped = mapped.replace(old, new)
                    shard = sid
                    break
            record(source, mapped, shard)
        else:
            record(source, mapped)
    loaded = namespace["load_weights"](model, iter(main_weights))
    for source, entry in catalog.items():
        if source not in weight_map:
            continue
        count = len(records.get(source, []))
        expected_count = 0 if entry["skip"] else 1
        if count != expected_count:
            issues.append(dict(kind="unexpected_load_count", key=source, actual=count, expected=expected_count))

    counts = Counter(catalog[k]["category"] for k in weight_map if k in catalog)
    samples = {}
    for layer in (0, 3, 44, 45):
        start = f"model.language_model.layers.{layer}."
        samples[str(layer)] = [dict(key=key, **catalog[key], header=metadata_by_key.get(key),
                                    actual_loader=records.get(key, []))
                               for key in catalog if key.startswith(start)
                               and (".experts." not in key or ".experts.0." in key)]
    # Missing shard files are an expected transport/inventory limitation in
    # this workspace (the user explicitly asked us not to handle scp). They
    # remain visible in ``unavailable_headers`` and do not invalidate the
    # mapping result for the keys whose headers are present.
    failed = bool(issues or missing_keys or unexpected_keys)
    status = "failed" if failed else (
        "passed_static_mapping_headers_partial_shards"
        if unavailable_shards or incomplete_payloads
        else "passed_static_mapping_and_headers"
    )
    return dict(
        status=status,
        scope="TP=EP=PP=1; no MTP; no model construction, tensor loading, numerical or DLC execution",
        mapping_method="Current Glm5NextModel.load_weights AST executed against independent config-derived recording parameters; outer/vision maps extracted from current source; BF16/AWQ helpers only",
        model_directory=str(model_dir), vllm_directory=str(vllm_dir),
        source_references=[str(model_source), str(vision_source),
                           str(root / "model_executor/models/glm4_1v.py"),
                           str(root / "model_executor/layers/fused_moe/routed_experts.py")],
        quantization_skip_modules={
            "raw_checkpoint_namespace": quant.get("modules_to_not_convert", []),
            "mapped_vllm_namespace": mapped_skip_modules,
            "catalog_key_match_counts": skip_match_counts,
            "mapper_source": "AutoAWQConfig.apply_vllm_mapper",
            "packed_mapping_current": current_packed_mapping,
            "packed_mapping_required_for_glm5_dense": required_packed_mapping,
            "dense_layer_skip_analysis": dense_skip_analysis,
        },
        counts=dict(index_keys=len(weight_map), expected_keys=len(catalog),
                    header_keys_checked=len(metadata_by_key), parameter_load_calls=sum(map(len, records.values())),
                    loaded_main_model_parameters=len(loaded), unique_target_slots=len(slots),
                    mtp_keys_skipped=counts["mtp_skipped"], by_category=dict(counts),
                    header_dtypes=dict(Counter(m["dtype"] for m in metadata_by_key.values()))),
        missing_expected_keys=missing_keys, unexpected_index_keys=unexpected_keys,
        issues=issues, unavailable_headers=unavailable_shards,
        unavailable_header_key_counts={
            shard: sum(1 for key, value in weight_map.items() if value == shard)
            for shard in (item["shard"] for item in unavailable_shards)
        },
        incomplete_payload_files=incomplete_payloads,
        limitations=["Parameter catalog is static; actual model construction and quant-method selection require separate tests.",
                     "FP8 dequantization, DLC physical layouts, tensor values, TP/EP/PP and MTP execution are not validated.",
                     "Payload availability is observed from file size only; checksum and payload contents are not read."],
        representative_layers=samples,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/data/models/GLM-5.3-Flash-Mix"))
    parser.add_argument("--vllm-dir", type=Path, default=Path("/root/workspace/vllm"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("GLM-5.3-Flash_weight_mapping_audit.json"))
    args = parser.parse_args()
    report = audit(args.model_dir, args.vllm_dir)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "counts")}, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")
    raise SystemExit(report["status"] == "failed")


if __name__ == "__main__":
    main()
