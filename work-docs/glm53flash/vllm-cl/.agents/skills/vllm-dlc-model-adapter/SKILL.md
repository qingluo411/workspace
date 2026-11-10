---
name: vllm-dlc-model-adapter
description: "Adapt and debug existing or new models for vLLM on DLC TPU. Confirm /vllm-workspace paths, avoid Triton and torch.compile paths, validate dummy first when useful, and require real-weight OpenAI-compatible inference before sign-off."
---

# vLLM DLC Model Adapter

## Overview

Adapt Hugging Face or local models to run on `vllm-dlc` with minimal code changes, deterministic validation, and compact Chinese handoff artifacts.

## Read order

1. Read `references/workflow-checklist.md`.
2. If startup, import, weight loading, or inference fails, read `references/troubleshooting.md`.
3. Before final response, read `references/deliverables.md`.

## When to use

Use this skill when:

- A user asks to adapt a model to vLLM DLC.
- A model starts but fails during DLC inference.
- A model requires DLC-specific patches for attention, MLA, MoE, distributed execution, quantization, or multimodal paths.
- A model path contains Triton or `torch.compile` assumptions that must be replaced for DLC.

## Hard constraints

- Confirm the implementation roots before changing code:
  - vLLM root defaults to `/vllm-workspace/vllm`.
  - vllm-dlc root defaults to `/vllm-workspace/vllm-dlc`.
- Start `vllm serve` from `/vllm-workspace/vllm` by default.
- Use default API port `8000` unless the user explicitly asks otherwise.
- Always use `--block-size 256` for DLC. Never omit it from startup commands.
- DLC does not support Triton kernels.
- DLC does not support `torch.compile` or `torch._dynamo` compiled paths.
- Replace Triton or compile-only logic with PyTorch native code or existing DLC-supported operators.
- Keep code changes minimal and focused on the target model.
- Dummy validation is useful but never sufficient for final sign-off.
- Real-weight validation with an OpenAI-compatible request is mandatory before claiming success.
- If the model supports multimodal input, validate at least one multimodal request.
- MTP is optional: after basic validation passes, ask the user whether to validate MTP.

## Quality gate before final answer

- `/vllm-workspace` path or replacement paths were confirmed.
- Every startup command used `--block-size 256`.
- Dummy stage evidence is reported if dummy was used.
- Real-weight stage evidence is reported and includes non-empty model output.
- `GET /v1/models` succeeded.
- `/v1/completions` succeeded with non-empty output.
- `/v1/chat/completions` succeeded with non-empty output.
- A long-prefix request exercised chunked prefill without errors.
- The server process remained alive after both short request paths and the long-prefix request.
- Multimodal request succeeded when the model supports multimodal input.
- Applicable DLC attention, MLA, MoE, or distributed paths were validated or marked not applicable with evidence.
- Logs or code review show Triton and `torch.compile` paths were not used.
- Test config or equivalent regression case exists.
- Final response is compact Chinese and includes changed files, commands, validation evidence, and known limits.
