# 历史 group128 临时接口测试说明

此说明归档自 kernel 仓的测试 README；临时 PyTorch 源码注册已撤回。设备命令需要临时接口或未来正式实现，重新编译已恢复的 PyTorch 源码本身不会提供该接口。

#### FP8 group128 quantization roundtrip / FP8 group128 量化往返

`tools/check_fp8_g128_roundtrip.py` checks the existing E4M3FN group128
quantization and BF16 dequantization through the PyTorch interfaces. Run from
`/root/workspace` with its Python environment when the loaded PyTorch build contains the temporary dequant API:

```bash
# Small first run: GLM latent+RoPE width, normal and outlier inputs.
uv run --no-sync --project /root/workspace /root/workspace/.venv/bin/python \
  work-tests/fp8_kvcache/check_fp8_g128_roundtrip.py \
  --device dlc:0 --shape 17,576 --patterns normal outliers

# Default shape/distribution cases, standalone dequantization and API checks.
uv run --no-sync --project /root/workspace /root/workspace/.venv/bin/python \
  work-tests/fp8_kvcache/check_fp8_g128_roundtrip.py \
  --device dlc:0 --json /tmp/fp8_g128_report.json

# CPU reference self-check; usable before rebuilding, never launches DLC work.
uv run --no-sync --project /root/workspace /root/workspace/.venv/bin/python \
  work-tests/fp8_kvcache/check_fp8_g128_roundtrip.py --cpu-only
```

The archived temporary interface is
`torch.dlc_dequant_e4m3(input, scale, output, group_size=128) -> None`.
It writes the caller-provided BF16 output. All tensors must be contiguous,
two-dimensional, on the same DLC device, and have zero storage offset:
input is E4M3FN `[T,D]`, scale is uint8 `[T,ceil(D/128)]`, and output is
BF16 `[T,D]` with separate storage. Empty inputs are no-ops. The scale codes
use the existing kernel convention `2**(code-127)`; this API does not introduce
group32 or claim MXFP8 special-code compatibility.

The script compares FP8 bytes, scale codes and restored BF16 against a CPU
reference, and separately reports quantization loss (max absolute error, RMSE,
cosine similarity and values rounded to zero). It also accepts
`--input sample.pt` for one saved CPU BF16 `[T,D]` tensor. It initializes FP8
test buffers on CPU and copies the same dtype to DLC, avoiding the unrelated
native FP8 `zeros/fill_` gap. Production conversion uses the two dedicated
device operators, not generic device FP8 casts. Rebuilding the kernel library
is unnecessary if the installed library already contains
`custom_dequant_e4m3_g128_bf16`.

