# Async D2H test suite

This directory implements the test plan in
`work-docs/async-d2h/test.md`. It is intentionally independent of any
specific model or serving configuration.

## Prerequisites

- The PyTorch target build is installed in the selected Python environment.
- Confirm the build before running:
  ```bash
  /root/workspace/.venv/bin/python - <<'PY'
  import torch
  print(torch.__version__)
  print(torch.version.git_version)
  print(torch.__file__)
  print(torch.dlc.is_available())
  PY
  ```
- Use real hardware for the async-contract checks. With `DLC_SYN_USE_SIM=1`,
  the simulator executes synchronously and the event-pending checks are
  skipped by the upstream test.
- Run from `/root/workspace`.

## A/B/C definitions

| Group | `DLC_SYN_COPY_ASYNC` | `DLC_DISABLE_DEVICE_D2H_LAYOUT` | Intended path |
|---|---:|---:|---|
| A | `O3` | `1` | O3 allocation, but forced synchronous fallback |
| B | `O3` | unset | New padded pinned + device-layout + async D2H path |
| C | `O2` | unset | Optional strategy comparison; not an async-D2H A/B |

Group A intentionally uses the fallback path. The upstream test's deferred
event assertion would be expected to fail there, so `run_existing_async_test.py`
skips **only** that assertion; all data, layout, fallback, stream, and lifetime
checks still run.

## Full suite

```bash
cd /root/workspace
/root/workspace/.venv/bin/python work-tests/async-d2h/run_ab_suite.py \
  --python /root/workspace/.venv/bin/python \
  --run-name after-build \
  --groups A,B \
  --repeats 10
```

Add `C` only when the O2 comparison is useful:

```bash
/root/workspace/.venv/bin/python work-tests/async-d2h/run_ab_suite.py \
  --python /root/workspace/.venv/bin/python \
  --run-name after-build-o2 \
  --groups A,B,C
```

Use `--dry-run` to inspect commands while the build is still running. Use
`--skip-existing`, `--skip-supplemental`, or `--skip-bench` for narrower runs.

### What the suite runs

1. `capture_env.py` for every group:
   - PyTorch version, `torch.version.git_version`, `torch.__file__`
   - DLC availability and relevant environment variables
   - runtime/driver files and `cltech0` sysfs version
2. Existing `pytorch/test/dlc_async_d2h_copy.py` for B/C.
   For A, `run_existing_async_test.py --fallback` runs the same test with only
   the event-pending assertion skipped.
3. `supplemental_async_d2h.py` for A/B:
   - basic shape/dtype/storage contract
   - slice/transpose fallback
   - controlled large-copy backlog
   - device allocation lifetime
   - pinned host allocation reuse/lifetime
   - non-default stream dependency
4. `bench_async_d2h.py` for A/B/C:
   - `.to()` submit time
   - event wait time
   - event-pending-at-return count
   - raw values, medians, p95, storage/pinned metadata

### Result layout

```text
work-tests/async-d2h/results/<run-name>/
├── summary.json
├── A/
│   ├── env.json
│   ├── existing.log
│   ├── supplemental.json
│   ├── micro.json
│   ├── micro.csv
│   └── *.log
└── B/
    └── ...
```

The summary is written only after all selected groups finish. Every command
and its return code are recorded; raw benchmark values are kept in
`micro.json` and `micro.csv`.

## Running one script manually

```bash
cd /root/workspace

# Group B: new path
DLC_SYN_COPY_ASYNC=O3 \
  /root/workspace/.venv/bin/python work-tests/async-d2h/supplemental_async_d2h.py \
  --mode async --rounds 32 --output /tmp/async-b.json

# Group A: fallback
DLC_SYN_COPY_ASYNC=O3 DLC_DISABLE_DEVICE_D2H_LAYOUT=1 \
  /root/workspace/.venv/bin/python work-tests/async-d2h/supplemental_async_d2h.py \
  --mode fallback --rounds 32 --output /tmp/async-a.json

# Microbenchmark
DLC_SYN_COPY_ASYNC=O3 \
  /root/workspace/.venv/bin/python work-tests/async-d2h/bench_async_d2h.py \
  --mode async --repeats 10 --output /tmp/micro-b.json
```

## vLLM end-to-end A/B

`run_vllm_ab.py` is client-side. It requires a model name and a running vLLM
service. Start the service separately for A and B, or pass `--server-cmd` so
the script starts and stops a fresh process for the selected group.

Example with an already-running service:

```bash
# Start the A server in another shell with:
# DLC_SYN_COPY_ASYNC=O3 DLC_DISABLE_DEVICE_D2H_LAYOUT=1 vllm serve ...
/root/workspace/.venv/bin/python work-tests/async-d2h/run_vllm_ab.py \
  --group A --base-url http://127.0.0.1:18005 \
  --model <served-model-name> --tokenizer <tokenizer-path> \
  --input-len 512 --output-len 128 --num-prompts 8 \
  --concurrencies 1 8 --repeats 3 \
  --run-name after-build

# Restart with group B environment, then run the same command with --group B.
```

Example with a server command launched by the script:

```bash
/root/workspace/.venv/bin/python work-tests/async-d2h/run_vllm_ab.py \
  --group B \
  --server-cmd 'vllm serve <model-path> --served-model-name <name> ...' \
  --base-url http://127.0.0.1:18005 \
  --model <name> --tokenizer <tokenizer-path> \
  --input-len 512 --output-len 128 --num-prompts 8 \
  --concurrencies 1 8 --repeats 3 \
  --run-name after-build
```

Use the same input/output lengths, prompt count, concurrency, dtype, parallel
configuration, scheduler configuration, and warmup for A and B. Do not use
different server flags between groups except the two D2H-related variables
under test.

For the timeline requirement, start the server with the vLLM profiler enabled,
for example:

```bash
vllm serve <model> ... \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/async-d2h-trace"}'
```

Then run the benchmark with `--profile`; the script will POST
`/start_profile` before the measured requests and `/stop_profile` afterward:

```bash
/root/workspace/.venv/bin/python work-tests/async-d2h/run_vllm_ab.py \
  --group B --profile --base-url http://127.0.0.1:18005 \
  --model <served-model-name> --tokenizer <tokenizer-path> \
  --input-len 512 --output-len 128 --num-prompts 8 \
  --concurrencies 1 8 --repeats 3 --run-name after-build-trace
```

Collect the torch trace and, if the repository's DLC profiler wrapper is
enabled, the corresponding `syn_trace` output. The timeline should show that
the worker submits the next batch while the previous sample D2H is still
pending, and that the event wait is in the async output thread.

## Qwen3-32B A/B wrapper

For the vLLM end-to-end smoke test, the repository-local wrapper uses
`/data/models/Qwen3-32B`, TP4, 4K context, and starts a fresh server for each
group:

```bash
cd /root/workspace
TEST_GROUPS="A B" REPEATS=3 NUM_PROMPTS=8 CONCURRENCIES="1 8" \
  ./work-tests/async-d2h/run_qwen3_32b_ab.sh
```

The wrapper keeps the server configuration identical between A and B except
for `DLC_SYN_COPY_ASYNC` / `DLC_DISABLE_DEVICE_D2H_LAYOUT`. The default server uses `--tensor-parallel-size 4`, `--max-model-len 4096`,
`--max-num-batched-tokens 4096`, `--max-num-seqs 8`, `--block-size 256`,
`--dtype bfloat16`, and `mode=none, cudagraph_mode=NONE` (non-eager but no
torch.compile/CUDAGraph). This keeps vLLM async scheduling enabled.

For this Qwen3 version, `FULL_DECODE_ONLY` currently enters a graph/compile
path that does not preserve the Qwen3 fused-QKV rotary dispatch and fails with
`RotaryEmbedding.forward_native() got an unexpected keyword argument 'qkv'`.
`ENFORCE_EAGER=1` works as a fallback, but vllm-cl intentionally disables
async scheduling for eager mode, so it is not the preferred D2H overlap test.
`COMPILATION_CONFIG` can be overridden once the Qwen3 graph path is fixed.

To capture a timeline, start with `PROFILE=1`, and configure the server with a
vLLM profiler directory in `run_qwen3_32b_ab.sh` or in your own
`--server-cmd`. The client-side script will POST `/start_profile` and
`/stop_profile` around the measured runs.

Current checkout caveat: the vllm-cl `patch_qwen3.py` in this workspace is
not compatible with the `FULL_DECODE_ONLY` graph path. Qwen3 works with
`mode=none + cudagraph_mode=NONE` and non-eager, with async scheduling enabled.
`--enforce-eager` also works, but disables async scheduling and is only a
serving smoke fallback. Fix the Qwen3 model patch before enabling graph mode.

## Interpreting results

- Correctness, fallback behavior, async submit, and lifetime checks are
  pass/fail gates.
- `pending_at_return` supports the async completion relation, but by itself
  does not prove compute/copy overlap.
- A shorter submit time does not imply the same reduction in data-ready time.
- End-to-end benefit is optional; do not preset a required percentage. Report
  raw TTFT/TPOT/ITL and throughput for A and B under identical conditions.
