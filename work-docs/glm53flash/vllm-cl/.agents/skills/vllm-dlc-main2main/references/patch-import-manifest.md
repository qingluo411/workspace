# Patch Import Manifest Reference

`vllm_cl/patch_import_manifest.yaml` records which upstream vLLM modules and symbols each vllm-dlc patch depends on.

## Schema

```yaml
patches:
  - dlc_file: vllm_cl/patch/models/patch_deepseek_v2.py
    vllm_modules:
      - vllm.model_executor.models.deepseek_v2
    watched_symbols:
      - DeepseekV2ForCausalLM
      - DeepseekV2Attention
```

## Rules

- `dlc_file` is the vllm-dlc patch file.
- `vllm_modules` lists upstream vLLM modules imported or monkey-patched by that file.
- `watched_symbols` lists upstream classes, functions, methods, or config symbols whose signature or behavior can break the patch.
- The manifest is an impact map, not a replacement for reading diffs.
- Import existence is not sufficient. During upgrade, inspect whether class structure, function signatures, forward parameters, config fields, or runtime behavior changed.
- Update the manifest whenever a patch starts depending on a new upstream module or symbol.
