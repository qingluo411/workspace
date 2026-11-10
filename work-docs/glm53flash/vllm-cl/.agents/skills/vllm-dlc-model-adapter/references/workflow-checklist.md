# vLLM DLC Model Adapter Workflow Checklist

## 1. Confirm context

Before changing code, confirm:

- vLLM root, default `/vllm-workspace/vllm`.
- vllm-dlc root, default `/vllm-workspace/vllm-dlc`.
- Whether `/vllm-workspace` exists; if not, ask for replacement paths.
- Model path.
- Delivery root.
- Whether the user wants a git commit.
- Whether the model is text-only, MoE, MLA, quantized, multimodal, or has MTP support.

## 2. Analyze the model

Inspect model files before choosing an implementation strategy:

- `config.json`
- tokenizer files
- processor files for multimodal models
- safetensors index and weight key prefixes
- architecture class names
- quantization config
- MTP-related modules or config fields

Decide whether the model can reuse an existing vLLM implementation, needs a vllm-dlc patch, or needs a DLC-supported replacement operator.

## 3. Choose dummy layer coverage

Dummy validation may reduce layer count with `hf_overrides`, but the number is not fixed. Read `config.json` first and choose a reduced layer count that still covers:

- embeddings
- attention
- MLA when present
- MLP or MoE
- normalization
- lm head
- model-specific special layers

`hf_overrides={"num_hidden_layers": 4}` is an example, not a universal rule.

## 4. Implement minimal adaptation

Prefer this order:

1. Reuse existing vLLM model implementations.
2. Patch targeted behavior under `vllm_dlc/patch`.
3. Use PyTorch native code for unsupported kernels.
4. Add or update `vllm_dlc/ops` only when PyTorch native is insufficient.

Do not use Triton kernels. Do not use `torch.compile`, `torch._dynamo`, or compile-only code paths.

## 5. Run Stage A dummy validation

Run from `/vllm-workspace/vllm`. Use dummy when useful, especially when real weights are unavailable or slow to load.

Default dummy parameters:

**Before running the command, explicitly check that `--block-size 256` is present.**

```bash
vllm serve --model <MODEL_PATH> \
  --load-format dummy \
  --enforce-eager \
  --dtype bfloat16 \
  --block-size 256 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  -tp 1
```

If using `hf_overrides`, include only the config keys justified by model analysis.

Required checks:

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<SERVED_MODEL_NAME>","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
```

Pass requires HTTP 200 and non-empty output. Startup alone is not enough.

## 6. Estimate real-weight TP

Before real-weight validation, estimate TP from model size and dtype. A single DLC card has about 60G usable memory, but reserve space for KV cache, activations, and framework overhead. Ask the user to confirm TP when the estimate is uncertain or when the user has a deployment target.

## 7. Run Stage B real-weight validation

Remove `--load-format dummy`. Use the confirmed or estimated TP.

**Before running the real-weight command, explicitly check that `--block-size 256` is present.**

Required checks:

- service readiness
- `GET /v1/models`
- one `/v1/completions` request with non-empty output
- one `/v1/chat/completions` request with non-empty output
- one long-prefix request that triggers chunked prefill without errors
- server process remains alive after both request paths
- no Triton path used
- no `torch.compile` path used

Use simple requests after the model is ready:

```bash
curl -s http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<SERVED_MODEL_NAME>","prompt":"Hello","max_tokens":5,"temperature":0}'

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<SERVED_MODEL_NAME>","messages":[{"role":"user","content":"Hello"}],"max_tokens":5,"temperature":0}'
```

Startup-only validation is not enough; a model that starts but crashes during
either short request path or the long-prefix chunked-prefill request has failed
Stage B. The long-prefix request should use enough prompt tokens to trigger
chunked prefill for the configured `max_model_len` while staying within the
model context limit.

If the model supports multimodal input, also send one text+image or corresponding multimodal request. If basic validation passes and MTP exists, ask the user whether to validate MTP.

## 8. Add regression coverage

Add a test config or equivalent regression case that records the model path, dtype, TP, fixed block size 256, and validation command. If repository test schema is unclear, create the smallest artifact that lets a future maintainer rerun the same smoke path.

## 9. Prepare Chinese handoff

Report:

- changed files
- model architecture and features
- dummy validation result
- real-weight validation result
- multimodal result when applicable
- whether MTP was tested or skipped by user choice
- exact startup commands
- exact request commands
- known limitations
- whether tutorial or commit was skipped because the user did not request it
