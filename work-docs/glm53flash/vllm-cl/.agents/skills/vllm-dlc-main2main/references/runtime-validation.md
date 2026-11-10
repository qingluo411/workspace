# Runtime Validation Reference

Use this reference after adapting code and whenever an upgrade touches attention, KV cache, dtype/quantization, model runners, sampling, distributed code, or CI configuration.

## 1. Validate in layers

Run the narrowest checks first, then expand:

1. `git diff --check` and Python syntax/import checks for changed modules.
2. Relevant `pytest` tests under `tests/` and kernel tests under `tests/kernels/`.
3. `ci_utils/ci_operation_test.sh` when an operator, dtype, layout conversion, or collective changed.
4. `ci_utils/ci_infer.sh` for normal infer coverage.
5. `ci_utils/tyd_ci_infer.sh` for models marked `is_valid_tyd`; this wrapper selects the TYD field and multi-device cases.
6. `ci_utils/ci_benchmark.sh` and `ci_utils/tyd_ci_benchmark.sh` when the change can affect performance or the user requests the performance gate.
7. Targeted real-weight smoke tests in both the default runner and V2 runner.

The infer scripts read `ci_utils/infer_models_list.json`. Respect each model's configured path, device count, block size, context length, memory utilization, eager/graph mode, expert parallel setting, compilation config, environment overrides, and HF overrides. Temporary CI model-path or selection edits are validation artifacts: keep them separate from implementation commits unless the user explicitly asks to retain them.

Run CI from its expected directory with the matching vLLM and vllm-dlc roots on `PYTHONPATH`. Record the exact command, selected model field, device list, log path, and exit code.

### Known scoped exception

The removed `compressed_tensors_to_dlc_rowpack` experiment has no corresponding operator in the current PyTorch/DLC build. Failures that depend only on that removed path in `tests/test_compressed_tensors_layout.py` may be reported as a known deferred exception for the current branch. Do not ignore unrelated failures, and do not restore or invent the operator merely to make those tests pass.

## 2. Representative real-weight matrix

At minimum, cover the model families affected by the diff. For a broad main-to-main upgrade, use all four:

| Model family | Typical scope | What it exercises |
| --- | --- | --- |
| Llama-2 7B | single device, bf16 | standard attention and `[B,H,N,2C]` KV layout |
| DeepSeek-V2 Lite | single device, bf16 | MLA cache and non-128 logical cache width |
| Qwen3 AWQ | four contiguous devices | quantized weights, TP collectives, standard KV layout |
| GLM-5.1 Mix | eight devices, TP/EP | large MoE/DSA model, multiple KV spec groups, physical padding and production memory pressure |

Resolve paths from the current CI configuration rather than assuming old mount points. In the current environment GLM-5.1 uses `/data/models/GLM-5.1-Mix/`. Its capacity gate uses `gpu_memory_utilization=0.97`; lowering it can hide a regression and is not equivalent validation. Use its configured 65,536-token context, expert-parallel and HF overrides when testing the CI-equivalent path.

For every representative model:

- confirm a non-empty, sensible output with a model-appropriate prompt
- confirm the process stays alive through prefill and decode
- verify chunked prefill where configured
- verify no Triton kernel and no `torch.compile`/`torch._dynamo` path is used
- capture model path, device list, TP/EP sizes, dtype, block size, memory utilization, context length, output, and exit code

Repeat the applicable matrix with:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

Require an explicit `Using V2 Model Runner` log. V2 validation must exercise a real request; import success, engine construction, profile-only runs, and empty dummy outputs are insufficient.

When V2 breaks after an upstream update, audit at least:

- KV-spec customization (`backend.customize_spec`) and cache binding contracts
- model-runner `prepare_inputs` and batch-state signatures
- attention metadata parameters and per-ubatch metadata builders
- index-mapping dtype expected by DLC fused operators
- sampler/logits-processing signatures and greedy output
- distributed communicator state used by enable/disable configuration

## 3. KV-cache upgrade gate

Do not rely only on a successful allocation. Inspect every distinct KV spec/layout group and representative layers in both the default and V2 runners.

### Runtime invariants

- Standard DLC fused KV cache remains `[B, H, N, 2C]` for the supported bf16/head-size/block-size path.
- Allocate the backend's physical cache shape directly. Do not reshape a generic upstream backing tensor into a logically convenient but physically incompatible view.
- Every non-shared attention layer gets an independent tensor with `storage_offset() == 0`. Only intentional KV-sharing layers may alias another layer.
- Record `shape`, `dtype`, `stride`, `storage_offset`, and `is_contiguous()` at attention forward, preferably once per distinct layout/worker to avoid log flooding.
- Validate standard attention, MLA, indexer, hybrid, and compressed groups separately when present; the first cache tensor alone may not represent the model.
- Verify `reshape_and_cache` and attention consume the intended tensor directly. A layout or offset mismatch can force `KernelDesc` materialization or a full KV-cache copy every forward.

### Physical padding and capacity

DLC pads the innermost physical dimension to a 512-byte multiple. For bf16 this is 256 elements; a logical width of 576 therefore occupies 768 elements. Memory planning must use the physical padded bytes for every independently allocated layer, not the logical `KVCacheSpec.page_size_bytes` and not a single packed-buffer assumption.

At the configured memory utilization, record:

- per-device model weight memory
- memory reported available for KV cache after profile run
- calculated number of cache blocks/tokens and maximum concurrency
- the physical size of the actual per-layer allocations

If allocation fails late, identify the exact allocation and its owner before calling it an overall OOM. The utilization snapshot can omit later per-layer padding or workspaces if the planner still uses logical sizes.

## 4. Diagnose unusually slow first inference

Large CI runs can launch several models, and each model's first weight load may take minutes. Do not use a universal one-minute timeout and do not classify unchanged console output alone as a hang.

Before terminating anything, inspect:

- checkpoint shard progress and filesystem type
- parent, EngineCore, and worker process state
- all per-rank/runtime logs, not only the parent log
- device occupancy and free HBM with `cltech_smi`
- whether the run is loading weights, profiling, allocating KV cache, warming kernels, or decoding

For multi-device tests use a healthy contiguous device group supported by the CI scripts, such as `0,1,2,3` or `4,5,6,7`. Clear only residual processes belonging to the current test. A single-device operator smoke can distinguish device health from model/KV problems.

If weight loading is complete but the first request is extremely slow, suspect an unintended KV-cache conversion or copy. Capture a short trace:

```bash
export DLC_SYN_PROF_CYCLE=1
export VLLM_TORCH_PROFILER_TOKENS=2

vllm serve <MODEL_AND_DLC_ARGS> \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir=/home/workspace/<TRACE_DIR> \
  --profiler-config.torch_profiler_with_stack=true

curl -sS -X POST http://127.0.0.1:8000/start_profile
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @<REQUEST_JSON>
curl -sS -X POST http://127.0.0.1:8000/stop_profile
```

Inspect `reshape_and_cache`, device-to-device copies, contiguous/materialization calls, and their stacks. Keep the profiler directory under `/home/workspace` unless the user chooses another location.

## 5. Dtype and quantization capability audit

For every new upstream dtype or quantization scheme:

1. Check whether the installed PyTorch exposes the dtype symbol.
2. Check whether DLC allocation, copy, attention, cache, and quantization operators actually support it.
3. Disable/comment unsupported registrations or selection branches so runtime cannot choose them.
4. If an import-time compatibility alias is necessary, keep it narrow and document that it does not add backend support. For example, mapping a missing `torch.float8_e8m0fnu` symbol to `torch.bfloat16` may unblock capability inspection, but all unsupported e8m0 execution paths must remain disabled.
5. Never infer support from a successful import or a patched symbol alone.

## 6. Result classification

Separate these outcomes in the report:

- functional pass: request completed with expected non-empty/sensible output
- capacity pass: production memory utilization and context length allocated successfully
- performance pass: no unexpected KV copy/materialization and benchmark threshold met when applicable
- known expected failure: exact test and already-scoped reason
- infrastructure/runtime issue: device state, missing model, or shutdown-only runtime crash

The DLC runtime may emit a known destructor/shutdown segfault after a completed request. Record it as a risk; do not confuse it with a request failure when output was produced and the controlling test exited successfully. Conversely, do not declare success from an initialization log if the request returned empty or workers died before decode.
