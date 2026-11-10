# vLLM DLC skills

This directory contains agent skills for vLLM DLC.

Copy `.agents/skills` to `.claude/skills` if you want these skills to be available as Claude Code slash commands in this repository.

## Skills

### vLLM DLC Model Adapter

Use `vllm-dlc-model-adapter` to adapt and debug models for vLLM on DLC TPU. The skill enforces DLC constraints such as fixed block size 256, no Triton kernels, no `torch.compile`, dummy-first validation, and mandatory real-weight validation.

### vLLM DLC main2main

Use `vllm-dlc-main2main` to adapt vllm-dlc to a newer vLLM version or commit. The skill reads `vllm_dlc/vllm_alignment.yaml`, checks patch dependencies in `vllm_dlc/patch_import_manifest.yaml`, and validates with real weights on the required local smoke models.
