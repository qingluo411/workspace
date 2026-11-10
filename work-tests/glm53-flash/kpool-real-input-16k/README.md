# GLM-5.3-Flash KPool real-input capture

This directory stores rank-0 inputs captured from real
`custom_vllm_kpool_select_topk` launches. The workload is one request with
exactly 16384 random prompt token IDs and 256 generated tokens.

## Workload and geometry

- Model: `/data/models/GLM-5.3-Flash-Mix`, BF16, TP=8.
- Service: V1 runner, `FULL_DECODE_ONLY` graph, `max_model_len=32768`,
  `max_num_batched_tokens=4096`, `max_num_seqs=1`, `block_size=256`.
- Request: random token IDs, seed `20261105`, `ignore_eos=true`,
  `min_tokens=max_tokens=256`; server usage reports 16384 prompt tokens and 256
  completion tokens.
- Chunked prefill creates four 4096-token chunks. The captured final chunk has
  `query=[4096, 32, 128]`, `seq_lens=[16384]`, `query_start_loc=[0, 4096]`.
- The 11 captures are the 11 sparse-MLA/indexer layers from that final chunk.
  They are not duplicate copies: each layer has different tensor contents and a
  different output SHA-256.

Each `capture_*/tensors.pt` contains CPU copies with the original dtypes/layout:

| key | shape | dtype |
|---|---|---|
| `query` | `[4096, 32, 128]` | `bfloat16` |
| `head_weights` | `[4096, 32]` | `float32` |
| `pool_cache` | `[1569, 64, 128]` | `bfloat16` |
| `block_table` | `[1, 128]` | `int32` |
| `query_start_loc` | `[2]` | `int32` |
| `seq_lens` | `[1]` | `int32` |
| `output` | `[4096, 2176]` | `int32` |

`output` is the exact result produced by the kernel during the real model run.
`metadata.json` records the operator scalars, shapes, context, block table
metadata and output hash. The complete capture is about 561 MiB.

## Capture

The opt-in hook is implemented in `vllm_cl/kpool_capture.py`; it is enabled only
when `DLC_KPOOL_CAPTURE_DIR` is set before the service starts. The launcher must
pass that environment to all workers. The configuration used here was:

```bash
cd /root/workspace/work-docs/glm53flash/vllm-cl/docs/glm5_flash
setsid env \
  PORT=18005 MAX_MODEL_LEN=32768 GPU_MEMORY_UTILIZATION=0.85 \
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
  DLC_KPOOL_CAPTURE_DIR=/root/workspace/work-tests/glm53-flash/kpool-real-input-16k \
  DLC_KPOOL_CAPTURE_DEVICE=0 \
  DLC_KPOOL_CAPTURE_MIN_TOKENS=2 \
  DLC_KPOOL_CAPTURE_MIN_SEQ_LEN=16384 \
  DLC_KPOOL_CAPTURE_LIMIT=11 \
  DLC_KPOOL_CAPTURE_TAG=real16k \
  bash ./start_glm5_flash_vllm_cl.sh \
    --max-num-batched-tokens 4096 --max-num-seqs 1 \
  > /root/workspace/log/kpool_capture_16k_service.log 2>&1 &
```

After `/health` returns 200, send the exact request:

```bash
/root/workspace/.venv/bin/python \
  /root/workspace/work-tests/glm53-flash/capture_kpool_real_input.py
```

The request took 149.51 s with capture enabled. The recorded 3.62 s per layer
covers the kernel wait plus synchronous D2H copy and serialization, so this
number is not a clean serving benchmark.

## Replay

Stop the model service first. Replay uses one TYD card and verifies the new
kernel output bit-for-bit against the captured real-model output:

```bash
/root/workspace/.venv/bin/python \
  /root/workspace/work-tests/glm53-flash/replay_kpool_select_topk.py
```

Replay one specific layer or change the iteration count:

```bash
/root/workspace/.venv/bin/python \
  /root/workspace/work-tests/glm53-flash/replay_kpool_select_topk.py \
  --bundle /root/workspace/work-tests/glm53-flash/kpool-real-input-16k/capture_0001_real16k_t4096_seq16384 \
  --warmup 1 --iterations 3
```

The current kernel passed all 11 layers with `mismatch_elements=0`. Median
device time was 3586.19 ms/layer (range 3586.08-3586.29 ms); the sum for 11
layers is 39.45 s. Full measurements are in `replay_results.json`.
