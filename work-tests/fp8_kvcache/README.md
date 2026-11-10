# FP8 KV Cache 预研测试

本目录不进入Git。所有命令从 `/root/workspace` 运行，使用工作区 `.venv`。
当前源码已正式提供 FP32-scale 反量化接口和 DLC FP8 初始化能力；分页 cache
writer/reader 仍不在本目录中实现。

- `fp8_reference.py`：新G128与group32数值合同的CPU golden，含tile pack/unpack、SAT/OVF、E8M0特殊值；legacy参考单独保留。
- `check_fp8_g128_roundtrip.py`：既有group128 legacy算法的数值/接口往返测试；CPU模式可独立运行，设备模式要求加载的PyTorch实际含临时 `torch.dlc_dequant_e4m3` 接口。
- `verify_reference.py`：早期KV方案的独立标量编码、布局和RMW检查，不启动DLC；直接运行会把结果写到本测试目录的 `reference_checks.json`，不覆盖历史文档附件。
- `check_fp32_scale_roundtrip.py`：正式 `torch.dlc_quant_e4m3` /
  `torch.dlc_dequant_e4m3` 的 FP32-scale 真机逐字节往返测试。
- `check_fp8_fill.py`：DLC 原生 FP8 `zeros/full/fill_/zero_`、NaN、切片和
  deterministic empty 测试，是 vLLM cache 分配前的阻塞检查。
- `check_dlc_fp8_layout.py`：用 `data_ptr()` 实测 FP8 值 cache 和 FP32
  scale 的物理行步长，防止按逻辑 nbytes 误判容量。
- `example_fp8_kv_cache.py`：`torch.dlc.fp8_kv_cache` 的可运行使用示例，
  对比 BF16、分离 FP8、`[slots,4]` scale 和 528B 内联行的真实显存分配，
  并演示量化写入、gather/dequant、负 slot、block copy 和 zero。
- `qwen3_kv_accuracy.py`：对运行中的 FP8/BF16 Qwen3 实例采集相同的贪心
  生成结果和 prompt-token logprobs，并生成自动对照 JSON。
- `qwen3_long_context.py`：构造 token 数精确的 8192-token needle prompt，
  记录完整输出、逐位置 prompt logprob 和耗时，可对 FP8/BF16 两次运行自动比较。
- `harness/test_dequant_fp8_g128_bf16.py`：增强版既有harness完整快照，含本次增加的负零/shape case。它依赖DLC_Custom_Kernel测试框架，若导入需将该仓根目录加入PYTHONPATH；其中E5接口原本未补齐，不把它作为独立可运行的全通过脚本。
- `patches/pytorch_temporary_dequant.patch`：临时PyTorch wrapper/schema/torchgen的完整撤回前补丁，供后续正式接入参考；未自动重新应用。
- `patches/kernel_test_changes.patch`：kernel测试README和既有dequant测试的本次增强补丁；仓库内原文件已恢复。

CPU检查：

```bash
uv run --no-sync --project /root/workspace /root/workspace/.venv/bin/python \
  work-tests/fp8_kvcache/fp8_reference.py --self-test
uv run --no-sync --project /root/workspace /root/workspace/.venv/bin/python \
  work-tests/fp8_kvcache/check_fp8_g128_roundtrip.py --cpu-only \
  --json work-tests/fp8_kvcache/reports/legacy_cpu_report.json
uv run --no-sync --project /root/workspace /root/workspace/.venv/bin/python \
  work-tests/fp8_kvcache/verify_reference.py
```

正式 FP32-scale 与 FP8 初始化设备测试：

```bash
/root/workspace/.venv/bin/python \
  work-tests/fp8_kvcache/check_fp32_scale_roundtrip.py --device dlc:0 \
  --json work-tests/fp8_kvcache/reports/fp32_scale_roundtrip_20261027.json
/root/workspace/.venv/bin/python \
  work-tests/fp8_kvcache/check_fp8_fill.py --device dlc:0 \
  --json work-tests/fp8_kvcache/reports/fp8_fill_20261027.json
/root/workspace/.venv/bin/python \
  work-tests/fp8_kvcache/check_dlc_fp8_layout.py --device dlc:0 \
  --json work-tests/fp8_kvcache/reports/fp8_layout_20261027.json
/root/workspace/.venv/bin/python \
  work-tests/fp8_kvcache/example_fp8_kv_cache.py --device dlc:0
```

Qwen3 TP4 低利用率容量/精度验证：

```bash
/root/workspace/.venv/bin/python work-tests/fp8_kvcache/qwen3_kv_accuracy.py \
  --model Qwen3-32B-Mix-fp8kv --label fp8_per_token_head \
  --output work-tests/fp8_kvcache/reports/qwen3_accuracy_fp8_lowutil_20261027.json
/root/workspace/.venv/bin/python work-tests/fp8_kvcache/qwen3_long_context.py \
  --model Qwen3-32B-Mix-fp8kv --label fp8_per_token_head_8192 --token-count 8192 \
  --output work-tests/fp8_kvcache/reports/qwen3_long8192_fp8_lowutil_20261027.json \
  --reference work-tests/fp8_kvcache/reports/qwen3_long8192_bf16_lowutil_20261027.json
```

本次低利用率结论见
[Qwen3 FP8 KV cache 验收报告](../../work-docs/fp8_kvcache/qwen3_fp8_kvcache_lowutil_validation_20261027.md)。

legacy `check_fp8_g128_roundtrip.py` 的 `--device` 模式仍只适用于包含对应
legacy dequant 入口的构建；CPU golden 不受影响。

[对应文档](../../work-docs/fp8_kvcache/README.md)和其中的
archive_manifest.json记录了原路径、源码HEAD及SHA256。最新实施契约见
[GLM-5.3 Sparse MLA FP8 实施契约](../../work-docs/fp8_kvcache/vllm-cl/glm53_sparse_mla_fp8_plan_20261027.md)。
当前报告只证明基础数值、PyTorch 接口和 FP8 初始化，分页 KV cache 写读和
vLLM 端到端仍未验收。
