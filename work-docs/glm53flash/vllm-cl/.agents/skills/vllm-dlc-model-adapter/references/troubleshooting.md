# vLLM DLC Model Adapter Troubleshooting

## Triton path is reached

Symptoms:

- import failure for `triton`
- a stack trace enters a Triton kernel
- model code expects a Triton attention, MoE, rotary, quantization, or sampling kernel

Action:

- Replace the path with PyTorch native code or an existing DLC-supported operator.
- Keep the replacement local to the affected model or patch.
- Re-run dummy validation and real-weight validation.

## torch.compile or dynamo path is reached

Symptoms:

- stack trace mentions `torch.compile`, `torch._dynamo`, compiled graph, or graph break handling
- code uses `@torch.compile` or wraps a function with `torch.compile(...)`

Action:

- Disable or bypass the compile path for DLC.
- Replace compile-only logic with eager PyTorch logic.
- Keep `--enforce-eager` in validation commands.
- Verify logs no longer enter the compile path.

## Dummy starts but real weights fail

Likely causes:

- weight key mismatch
- quantization path mismatch
- tensor parallel sharding issue
- real tensor shape differs from dummy assumptions

Action:

- Inspect safetensors index and actual key prefixes.
- Add explicit weight mapping when needed.
- Re-estimate TP using model size and dtype.
- Re-run real-weight validation; do not sign off with dummy-only evidence.

## Service starts but first request crashes

Likely causes:

- lazy model path only runs during first request
- sampler or logits processor issue
- attention or KV cache issue

Action:

- Treat startup as incomplete validation.
- Use the first request traceback as the root cause.
- Fix the runtime path and rerun `/v1/models` plus chat completion.

## Multimodal processor fails

Likely causes:

- processor API mismatch
- missing image/video token handling
- visual encoder path uses unsupported kernels

Action:

- Separate text-only validation from multimodal validation.
- Inspect processor config and model multimodal fields.
- Replace unsupported Triton or compile paths in visual modules.
- Final sign-off requires multimodal request if the model supports multimodal input.

## MTP path is present

Action:

- Do not make MTP part of the default gate.
- After basic real-weight validation passes, ask the user whether to validate MTP.
- If the user requests MTP validation, test the MTP-specific proposer, sampling, and weight-loading path.
