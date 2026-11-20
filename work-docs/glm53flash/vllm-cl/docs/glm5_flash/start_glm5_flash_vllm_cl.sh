#!/usr/bin/env bash
set -euo pipefail

# GLM-5.3-Flash production launcher.
# Decode graph capture is the default (FULL_DECODE_ONLY); prefill and chunked
# prefill stay eager.  Use `--eager` only for triage or output comparisons.
#
# Optional launcher switches (everything else is forwarded to `vllm serve`):
#   --graph                accepted for compatibility (graph is already default)
#   --eager                disable graph capture (--enforce-eager)
#   --profile              enable the vLLM torch profiler on the server
#   --profile=<dir>        same, with an explicit trace output directory
#   --kernel-log           write the Synapse per-launch kernel log (tensor
#                          shapes/strides/dtypes/scalars + device cycles)
#   --kernel-log=<dir>     same, with an explicit output directory
#   --prompt-token-details forward --enable-prompt-tokens-details so OpenAI
#                          responses include usage.prompt_tokens_details.cached_tokens
#                          (opt-in; useful for prefix-cache hit validation)
#
# `--kernel-log` exports DLC_SYN_LOG_DIR / DLC_SYN_VERBOSE=1 / DLC_SYN_DEBUG=1
# / DLC_SYN_PROF_CYCLE=1; each process then writes <prefix>_<pid>.ansi holding
# one "launch  <name> ... DLCTensor[...] shape=[...] stride=[...] ... cycles: N
# ops: N bytes: N crt: N" block per kernel launch.  Turn those into a
# per-(kernel, args) table with:
#   python /root/workspace/llama2-fine-tune/tool.py <dir>   # -> *_kernels.txt
# It is very verbose (one block per launch, per rank), so use it for short runs.
# The service still starts and serves normally; a trace window is taken on
# demand from another shell:
#   curl -sS -X POST http://127.0.0.1:$PORT/start_profile
#   curl -sS http://127.0.0.1:$PORT/v1/chat/completions \
#        -H 'Content-Type: application/json' \
#        --data-binary @/tmp/request_body_from_test_2048_3.json
#   curl -sS -X POST http://127.0.0.1:$PORT/stop_profile
# Traces land in $GLM5_PROFILER_DIR (default /tmp/syn_log) as *.pt.trace.json.gz
# and can be opened with https://ui.perfetto.dev/.
# Profiling mode also exports VLLM_TORCH_PROFILER_TOKENS (how many model steps
# to record after /start_profile, default 5; 0 would mean "until
# /stop_profile"), raises VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS to 3600 (tracing
# can slow a step past the 300 s default, which kills the engine core), and
# forces DLC_SYN_PROF_CYCLE=1 (DLC cycle counters), overriding the
# DLC_SYN_PROF_CYCLE=0 that the dev shell exports.

MODEL_PATH="${GLM5_MODEL:-/data/models/GLM-5.3-Flash-Mix}"
VLLM_ROOT="${VLLM_ROOT:-/root/workspace/vllm}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.3-Flash-Mix}"
PORT="${PORT:-8000}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
VLLM_BIN="${VLLM_BIN:-/root/workspace/.venv/bin/vllm}"
# The checkpoint advertises a 1M-token context.  A single 8x TYD node cannot
# hold that KV cache at the default memory utilization, so the validation
# default is the documented 128K bring-up target.  Override with MAX_MODEL_LEN
# (for example MAX_MODEL_LEN=193536) after checking the available KV memory.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

# vLLM starts a model-registry subprocess, so compatibility stubs must be
# visible to every Python process, not only the launcher.  The stubs only
# affect optional multimodal/media imports; GLM-5.3 text inference does not
# use them.  Set GLM5_VLLM_MEDIA_STUBS=0 when the installed wheels are valid.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPAT_DIR="$SCRIPT_DIR/compat"
if [[ -d "$COMPAT_DIR" ]]; then
  export PYTHONPATH="$COMPAT_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

# Driver v26.0.1 matches DLCSynapse Api_Version_26 and is the validated
# production combination.  Keep the check strict by default; only set
# DLC_SKIP_VERSION_CHECK=1 for isolated old test images, never for production.
export DLC_SKIP_VERSION_CHECK="${DLC_SKIP_VERSION_CHECK:-0}"

# Synapse graph replay cache. This is not required to capture a graph, but
# without it every replay walks graph nodes instead of reusing the recorded
# kernel-queue batches. Keep the validated production path enabled by default;
# set DLC_SYN_GRAPH_CACHE=0 only for cache-specific triage.
export DLC_SYN_GRAPH_CACHE="${DLC_SYN_GRAPH_CACHE:-1}"

# DLC graph capture can race with non-blocking CPU->device metadata copies
# (notably query_start_loc in MTP draft metadata). Use the PyTorch DLC
# async-copy level switch to make H2D synchronous in the service.
# O0=synchronous, O3=default async; see
# pytorch/c10/dlc/DLCConstants.h.
export DLC_SYN_COPY_ASYNC="${DLC_SYN_COPY_ASYNC:-O0}"

# Load the DLC platform plugin and its general patch hook.  The allowlist must
# include both names: `dlc` selects the platform and `dlc_patches` installs
# vllm-cl's model/KV-cache patches in the API, engine-core, and worker
# processes.  Loading only `dlc` leaves the engine core on upstream code.
export VLLM_PLUGINS="${VLLM_PLUGINS:-dlc,dlc_patches}"

# `--profile` is consumed here so it is not forwarded to `vllm serve`.
PROFILE="${GLM5_PROFILE:-0}"
PROFILER_DIR="${GLM5_PROFILER_DIR:-/tmp/syn_log}"
KERNEL_LOG="${GLM5_KERNEL_LOG:-0}"
SYN_LOG_DIR="${GLM5_SYN_LOG_DIR:-/root/workspace/log/syn_log}"
GRAPH="${GLM5_GRAPH:-1}"
PROMPT_TOKEN_DETAILS="${GLM5_PROMPT_TOKEN_DETAILS:-0}"
SERVE_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --graph) GRAPH=1 ;;
    --eager) GRAPH=0 ;;
    --profile) PROFILE=1 ;;
    --profile=*) PROFILE=1; PROFILER_DIR="${arg#--profile=}" ;;
    --kernel-log) KERNEL_LOG=1 ;;
    --kernel-log=*) KERNEL_LOG=1; SYN_LOG_DIR="${arg#--kernel-log=}" ;;
    --prompt-token-details|--enable-prompt-tokens-details) PROMPT_TOKEN_DETAILS=1 ;;
    *) SERVE_ARGS+=("$arg") ;;
  esac
done

case "$PROMPT_TOKEN_DETAILS" in
  1|true|True|TRUE|on|ON|yes|YES)
    SERVE_ARGS+=(--enable-prompt-tokens-details)
    echo "[metrics] prompt token details enabled; usage.prompt_tokens_details.cached_tokens will be populated" >&2
    ;;
esac

case "$GRAPH" in
  1|true|True|TRUE|on|ON|yes|YES) GRAPH=1 ;;
  *) GRAPH=0 ;;
esac
GRAPH_COMPILATION_ARGS=()
if [[ "$GRAPH" == "1" ]]; then
  # The DLC build replaces torch.compile with a pass-through.  Piecewise
  # graphs therefore have no compiled subgraphs to capture; FULL_DECODE_ONLY
  # gives us the useful uniform-decode graph while prefill stays eager.
  # Override with GLM5_CUDAGRAPH_MODE after validating another mode.
  GRAPH_COMPILATION_ARGS=(
    --compilation-config "{\"cudagraph_mode\": \"${GLM5_CUDAGRAPH_MODE:-FULL_DECODE_ONLY}\"}"
    --kernel-config '{"enable_jit_warmup": false}'
  )
  echo "[graph] decode graph capture enabled; mode=${GLM5_CUDAGRAPH_MODE:-FULL_DECODE_ONLY}; JIT warmup disabled; prefill remains eager" >&2
else
  echo "[graph] disabled; server runs fully eager" >&2
fi

# vLLM reads VLLM_LOGGING_LEVEL once at import time; the dev shell exports
# DEBUG, which floods every process log.  Default this launcher to INFO and
# allow opting back into DEBUG with GLM5_LOG_LEVEL=DEBUG.
export VLLM_LOGGING_LEVEL="${GLM5_LOG_LEVEL:-INFO}"

# The DLC linear loader uses this layout for the adapted checkpoint path.
# Keep an existing caller-provided value; default to the DLC path used by the
# validation scripts.
export VLLM_USE_DLC_COL_MAJOR_MATMUL="${VLLM_USE_DLC_COL_MAJOR_MATMUL:-1}"

# Mamba (34 x ~673 MiB) and MLA tensors are allocated interleaved, which
# fragments the caching allocator and fails large contiguous mamba pages
# (see the PYTORCH_DLC_ALLOC_CONF hint in the OOM message). Expandable
# segments avoid the contiguous-block requirement.
export PYTORCH_DLC_ALLOC_CONF="${PYTORCH_DLC_ALLOC_CONF:-expandable_segments:True}"

# Synapse kernel-detail log: one block per kernel launch, carrying the tensor
# descriptors (shape/stride/dtype/offset) and, with PROF_CYCLE, the device
# cycles/ops/bytes/crt footer.  DEBUG + PROF_CYCLE gate the block (see
# DLCSynapse/synapse_core/synapse_core.cpp:2204); VERBOSE=1 additionally prints
# the submit names.  Parse with llama2-fine-tune/tool.py -> *_kernels.txt.
SYN_LOG_PREFIX="${GLM5_SYN_LOG_PREFIX:-syn}"
if [[ "$KERNEL_LOG" != "0" && "$KERNEL_LOG" != "false" && "$KERNEL_LOG" != "off" ]]; then
  export DLC_SYN_LOG_DIR="$SYN_LOG_DIR"
  export DLC_SYN_LOG_PREFIX="$SYN_LOG_PREFIX"
  export DLC_SYN_VERBOSE="${GLM5_SYN_VERBOSE:-1}"
  export DLC_SYN_DEBUG=1
  # The dev shell exports DLC_SYN_PROF_CYCLE=0, which would drop the
  # cycles/ops/bytes/crt footer from every block.
  export DLC_SYN_PROF_CYCLE=1
  mkdir -p "$DLC_SYN_LOG_DIR"
  echo "[kernel-log] DLC_SYN_LOG_DIR=$DLC_SYN_LOG_DIR prefix=$DLC_SYN_LOG_PREFIX VERBOSE=$DLC_SYN_VERBOSE DEBUG=$DLC_SYN_DEBUG PROF_CYCLE=$DLC_SYN_PROF_CYCLE" >&2
  echo "[kernel-log] one <prefix>_<pid>.ansi per process; parse with:" >&2
  echo "[kernel-log]   ${GLM5_TOOL_PY:-/root/workspace/llama2-fine-tune/tool.py} $DLC_SYN_LOG_DIR" >&2
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "model directory does not exist: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -d "$VLLM_ROOT" ]]; then
  echo "vLLM root does not exist: $VLLM_ROOT" >&2
  exit 2
fi
if ! command -v "$VLLM_BIN" >/dev/null 2>&1; then
  echo "vLLM executable not found: $VLLM_BIN" >&2
  echo "set VLLM_BIN to the vLLM executable in the DLC environment" >&2
  exit 2
fi

PROFILER_ARGS=()
if [[ "$PROFILE" != "0" && "$PROFILE" != "false" && "$PROFILE" != "off" ]]; then
  # VLLM_TORCH_PROFILER_TOKENS caps how many model steps are recorded after
  # /start_profile (0 = record until /stop_profile).  DLC cycle counters come
  # from the runtime and are enabled through the environment before Synapse
  # initialises.
  export VLLM_TORCH_PROFILER_TOKENS="${VLLM_TORCH_PROFILER_TOKENS:-5}"
  # Tracing multiplies step time; the 300 s default for the engine-core RPC
  # (execute_model) would abort the engine before even one traced step lands.
  export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${GLM5_EXECUTE_MODEL_TIMEOUT:-3600}"
  # The dev shell exports DLC_SYN_PROF_CYCLE=0, which would silently disable the
  # DLC cycle counters; profile mode forces it on (GLM5_PROF_CYCLE=0 to opt out).
  export DLC_SYN_PROF_CYCLE="${GLM5_PROF_CYCLE:-1}"
  # The torch trace only carries host-side time ("Self DLC" stays 0, see the
  # perf plan P2-1).  The runtime writes its own per-thread device trace --
  # syn_trace_<tid>.json, flushed every 5 s -- into DLC_SYN_PROF_DIR, and with
  # PROF_CYCLE=1 each record carries device cycles.  Put it next to the torch
  # traces; GLM5_SYN_PROF_DIR overrides, GLM5_SYN_PROF_DIR=0 disables.
  if [[ "${GLM5_SYN_PROF_DIR:-1}" != "0" ]]; then
    export DLC_SYN_PROF_DIR="${GLM5_SYN_PROF_DIR:-$PROFILER_DIR/syn_trace}"
  fi
  # Per-kernel device cycles additionally need the device-trace HBM buffer
  # reserved before Synapse initialises; it is off by default because it
  # changes the init-time memory layout, so make it opt-in.
  if [[ "${GLM5_DEVICE_TRACE:-0}" == "1" ]]; then
    export DLC_SYN_DEVICE_TRACE=1
  fi
  mkdir -p "$PROFILER_DIR"
  [[ -n "${DLC_SYN_PROF_DIR:-}" ]] && mkdir -p "$DLC_SYN_PROF_DIR"
  echo "[profile] vLLM torch profiler enabled: VLLM_TORCH_PROFILER_TOKENS=$VLLM_TORCH_PROFILER_TOKENS DLC_SYN_PROF_CYCLE=$DLC_SYN_PROF_CYCLE DLC_SYN_PROF_DIR=${DLC_SYN_PROF_DIR:-<unset>} DLC_SYN_DEVICE_TRACE=${DLC_SYN_DEVICE_TRACE:-0} VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS" >&2
  echo "[profile] traces -> $PROFILER_DIR (open the *.pt.trace.json.gz with https://ui.perfetto.dev/)" >&2
  echo "[profile] after 'Application startup complete', record a window with:" >&2
  echo "[profile]   curl -sS -X POST http://127.0.0.1:$PORT/start_profile" >&2
  echo "[profile]   curl -sS http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' --data-binary @/tmp/request_body_from_test_2048_3.json" >&2
  echo "[profile]   curl -sS -X POST http://127.0.0.1:$PORT/stop_profile" >&2
  echo "[profile] tip: profile a tiny request first (/tmp/req_tiny_chat.json, max_tokens=2)." >&2
  PROFILER_ARGS=(
    --profiler-config.profiler=torch
    "--profiler-config.torch_profiler_dir=$PROFILER_DIR"
    --profiler-config.torch_profiler_with_stack=true
  )
fi

# Either profiling mode turns on DLCCL's explicit pre-collective sync kernel, so
# the trace can split "waiting for the other ranks" from the actual RDMA
# traffic: a long sync kernel with a short collective means the cost is
# rendezvous latency, not bandwidth.  See DLC_CL/README.md "Sync-All-Machine
# Kernel"; GLM5_DLCCL_SYNC_ALL_MACHINE=0 opts out.
if [[ "$PROFILE" != "0" && "$PROFILE" != "false" && "$PROFILE" != "off" ]] ||
   [[ "$KERNEL_LOG" != "0" && "$KERNEL_LOG" != "false" && "$KERNEL_LOG" != "off" ]]; then
  export DLCCL_SYNC_ALL_MACHINE="${GLM5_DLCCL_SYNC_ALL_MACHINE:-1}"
  echo "[dlccl] DLCCL_SYNC_ALL_MACHINE=$DLCCL_SYNC_ALL_MACHINE (pre-collective sync kernel split out for profiling)" >&2
fi

cd "$VLLM_ROOT"
SERVE_CMD=(
  "$VLLM_BIN" serve "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --language-model-only
  --dtype bfloat16
  --max-model-len "$MAX_MODEL_LEN"
  --block-size 256
  --tensor-parallel-size "$TP_SIZE"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
  --trust-remote-code
  --port "$PORT"
  --enable-logging-iteration-details
)
if [[ "$GRAPH" != "1" ]]; then
  SERVE_CMD+=(--enforce-eager)
elif [[ ${#GRAPH_COMPILATION_ARGS[@]} -gt 0 ]]; then
  SERVE_CMD+=("${GRAPH_COMPILATION_ARGS[@]}")
fi
if [[ ${#PROFILER_ARGS[@]} -gt 0 ]]; then
  SERVE_CMD+=("${PROFILER_ARGS[@]}")
fi
if [[ ${#SERVE_ARGS[@]} -gt 0 ]]; then
  SERVE_CMD+=("${SERVE_ARGS[@]}")
fi

exec "${SERVE_CMD[@]}"
