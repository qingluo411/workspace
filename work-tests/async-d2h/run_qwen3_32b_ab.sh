#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/root/workspace}
PYTHON=${PYTHON:-/root/workspace/.venv/bin/python}
VLLM_BIN=${VLLM_BIN:-/root/workspace/.venv/bin/vllm}
MODEL_PATH=${MODEL_PATH:-/data/models/Qwen3-32B}
MODEL_NAME=${MODEL_NAME:-Qwen3-32B}
PORT=${PORT:-18005}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT}}
TP_SIZE=${TP_SIZE:-4}
TEST_GROUPS=${TEST_GROUPS:-"A B"}
REPEATS=${REPEATS:-3}
NUM_PROMPTS=${NUM_PROMPTS:-8}
CONCURRENCIES=${CONCURRENCIES:-"1 8"}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
RUN_NAME=${RUN_NAME:-qwen3-32b-async-d2h-$(date +%Y%m%d-%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-${WORKSPACE}/work-tests/async-d2h/results}
VLLM_CWD=${VLLM_CWD:-${WORKSPACE}/vllm}
PROFILE=${PROFILE:-0}
PROFILER_CONFIG=${PROFILER_CONFIG:-}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"mode":"none","cudagraph_mode":"NONE"}'}

export VLLM_PLUGINS=${VLLM_PLUGINS:-dlc,dlc_patches}
export DLC_SKIP_VERSION_CHECK=${DLC_SKIP_VERSION_CHECK:-0}
export DLC_SYN_GRAPH_CACHE=${DLC_SYN_GRAPH_CACHE:-1}
export PYTORCH_DLC_ALLOC_CONF=${PYTORCH_DLC_ALLOC_CONF:-expandable_segments:True}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

EAGER_FLAG=""
PROFILER_ARG=""
if [[ -n "${PROFILER_CONFIG}" ]]; then
  PROFILER_ARG="--profiler-config '${PROFILER_CONFIG}'"
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  EAGER_FLAG="--enforce-eager"
fi

SERVER_CMD="${VLLM_BIN} serve '${MODEL_PATH}' \
  --served-model-name '${MODEL_NAME}' \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --block-size 256 \
  --tensor-parallel-size ${TP_SIZE} \
  --gpu-memory-utilization 0.80 \
  --trust-remote-code \
  ${EAGER_FLAG} \
  --port ${PORT} \
  --compilation-config '${COMPILATION_CONFIG}' \
  ${PROFILER_ARG}"

PROFILE_ARGS=()
if [[ "${PROFILE}" == "1" ]]; then
  PROFILE_ARGS+=(--profile)
fi

for GROUP in ${TEST_GROUPS}; do
  echo "=== Qwen3-32B group ${GROUP} ==="
  "${PYTHON}" "${WORKSPACE}/work-tests/async-d2h/run_vllm_ab.py" \
    --group "${GROUP}" \
    --vllm-cwd "${VLLM_CWD}" \
    --server-cmd "${SERVER_CMD}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_NAME}" \
    --tokenizer "${MODEL_PATH}" \
    --input-len "${INPUT_LEN}" \
    --output-len "${OUTPUT_LEN}" \
    --num-prompts "${NUM_PROMPTS}" \
    --concurrencies ${CONCURRENCIES} \
    --repeats "${REPEATS}" \
    --warmup-runs 1 \
    --result-root "${RESULT_ROOT}" \
    --run-name "${RUN_NAME}" \
    "${PROFILE_ARGS[@]}"
done
