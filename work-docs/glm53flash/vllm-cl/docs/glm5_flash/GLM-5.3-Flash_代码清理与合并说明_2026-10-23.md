# GLM-5.3-Flash 代码清理与合并说明（2026-10-23）

## 1. 目标与结论

本次整理面向 GLM-5.3-Flash 的正式发布，目标是把特性分支上的适配与优化收敛为可合入主分支的生产实现：

- 以 `FULL_DECODE_ONLY` graph 捕获作为默认配置，prefill 与 chunked prefill 保持 eager。
- 删除 GLM 生产路径中的 native/fallback 执行路线、CPU 对比分发框架和验证专用探针。
- 保留必要的算子契约测试与生产所需的 CPU 辅助函数，统一格式化并清理死代码。
- 保持 GLM-5.3-Flash 的功能、KPool 融合算子、KDA、MHC 和 sparse MLA 路径完整可用。

当前整理结果：vllm-cl 定向测试 `161 passed`，vllm 新增 graph padding 用例以独立等价脚本验证通过；发布 smoke 通过单请求、8 并发和 3610-token 长上下文验证。

## 2. 仓库与分支状态

| 仓库 | 当前分支 | 说明 |
|---|---|---|
| `vllm` | `release-glm53flash` | 基于 `upstream-v0.30.1rc0` 的 GLM 适配；工作区含 KDA graph padding 修正 |
| `vllm-cl` | `release-glm53flash` | GLM 插件、算子接入、缓存布局与 graph 支持的主要实现 |
| 文档 | `/root/workspace/work-docs/glm53flash` | 所有设计、需求、验证与交付文档只归档在这里，不进入 vllm-cl |

当前改动尚未提交。正式提交前建议按仓库分别组织提交：

1. vllm-cl：GLM 功能迁移、KPool/KDA/MHC/sparse MLA 接入、graph 支持。
2. vllm-cl：native/调试代码清理、格式化与测试收敛。
3. vllm：KDA graph padding 与 metadata 边界修正。
4. work-docs：仅归档文档，不纳入 vllm-cl 发布分支。

## 3. 主要代码改动

### 3.1 GLM-5.3-Flash 模型接入

- `vllm_cl/patch/models/patch_glm5next.py`：安装 GLM5Next 模型、indexer、稀疏 attention 与缓存相关 patch。
- `vllm_cl/patch/models/glm5_indexer.py`：接入 KPool indexer，统一 `glm5_kpool` 缓存版本、池大小、TopK 宽度和融合调用。
- `vllm_cl/attention/sparse_mla.py`：DLC sparse MLA backend 收敛为 `DLCMLASparse*` / `DLC_MLA_SPARSE_VLLM_V1`，支持 NoPE-512 路径、graph uniform batch 和直接复用 TopK 输出。
- `vllm_cl/ops/sparse_attn_indexer.py`、`patch_mla_indexer.py`：接入 indexer K cache 更新、TopK 和 sparse attention 相关 DLC 算子。

### 3.2 KDA 与状态池

- `vllm_cl/patch/patch_kda.py`：接入 KDA causal conv prefill/update、chunk prefill、recurrent decode、state gather/scatter 和 RMSNorm sigmoid gate。
- `vllm_cl/worker/mamba_cache.py`、`patch_kv_cache_interface.py`：支持 KDA 与 Mamba 状态缓存的分组、页布局和容量计算。
- `vllm/models/glm5next/common/kda.py`（vllm）：#04b 改为使用完整 persistent `cu_seqlens`；graph 用 PAD 槽补齐时，不能再截断到 `num_decodes`。
- `vllm/v1/attention/backends/gdn_attn.py`（vllm）：FULL graph decode 下为尾部 padding 请求补齐 cumulative boundaries，并保持 NULL state 槽为 `-1`。

### 3.3 KPool 融合路径

- `vllm_cl/patch/models/kpool_indexer.py`：#07 append+compress 与 #09 select+expand 的生产调用路径；删除旧的 Python 编排和多段调度代码。
- `vllm_cl/kpool_worker_slots.py`：#19 slot mapping 的融合缓存按完整容量分配，padding 槽填 `-1`，供 graph capture 与 replay 复用。
- `vllm_cl/_custom_ops.py`：生产 wrapper 直接调用融合算子；#09 输出要求 128 列对齐的 contiguous int32 持久缓冲区。
- `vllm_cl/patch/patch_kpool_cache_manager.py`、`patch_kv_cache_utils.py`：KPool page 的物理大小、分组和 KV 容量估算。
- KPool speculative transaction 已删除，不再保留 `kpool_speculative.py` 及其 runner 分支。

### 3.4 MHC、激活与量化

- `vllm_cl/patch/patch_mhc.py`：接入 `mhc_pre`、`mhc_post` 和 fused post+pre 路径，子层 RMSNorm 在融合 kernel 内完成。
- `vllm_cl/ops/activation.py`：接入 `silu_and_mul_with_clamp`，生产路径不再保留 unfused 分支。
- `vllm_cl/ops/awq_marlin.py`、`_custom_ops.py`：保留 AWQ MoE CPU fallback 实际需要的布局逆变换和反量化辅助函数；删除仅用于调试的 CPU/DLC 对比分发框架。

### 3.5 KV cache、metadata 与 graph

- `vllm_cl/worker/metadata_packer.py`：合并 metadata H2D，减少逐层标量和重复小拷贝。
- `vllm_cl/patch/patch_block_table.py`、`patch_kv_block_zeroer.py`：KPool/Mamba page 的 block table、零块初始化与 padding 处理。
- `vllm_cl/worker/dlc_model_runner.py`：在 profile/capture 前显式生成 #19 KPool slots；graph decode 复用持久 KPool 输出；修正 padded batch 的 indexer 输出宽度。
- `vllm_cl/patch/models/glm5_indexer.py`：padding batch 使用 `out=out[:actual_tokens]`，避免把 graph PAD token 的 top-k 结果误判为真实宽度。
- `vllm_cl/platform.py`、`envs.py`：收敛 DLC 平台配置和环境变量，删除 `VLLM_ENABLE_DLC_DEBUG`。

## 4. 本次删除与收敛

### 4.1 删除的生产外路径

- `vllm_cl/kpool_speculative.py`：native KPool speculative transaction 已无生产调用，删除。
- `vllm_cl/patch/models/kpool_slot_mapping.py`：旧 Python slot mapping 路线，由 #19 融合算子替代。
- `vllm_cl/patch/kda_state_pool_dlc.py`：旧 KDA state pool 实现，由当前 KDA/state pool 路径替代。
- `PyTorchMLASparse*` 旧实现、未被调用的 forward 和大量注释代码，从 sparse MLA 中删除。
- `_custom_ops.py` 中的 `Dispatch`、`OP_STATS`、`process_op`、CPU/DLC 双跑对比和 `flash_attn_varlen_func_unfused` 删除。

### 4.2 删除的 native/验证测试

- `tests/test_kda_native.py`
- `tests/test_kpool_indexer_native.py`
- `tests/test_kpool_metadata_native.py`
- `tests/test_kpool_slot_mapping_native.py`
- `tests/test_kpool_speculative.py`
- `tests/test_sparse_mla_index_conversion.py`

保留的是生产契约、布局、缓存容量、算子 ABI 和 GLM 集成测试；上游协议名 `forward_native`、`_supports_native_decode` 等如需接口兼容可以保留，但它们不再代表 GLM 的 native 执行路线。

### 4.3 命名统一

- `is_native_kpool_spec` → `is_kpool_spec`
- `native_kpool_cache_shape` → `kpool_cache_shape`
- `native_kpool_page_bytes` → `kpool_page_bytes`
- `NATIVE_KPOOL_MODEL_VERSION` → `KPOOL_MODEL_VERSION`
- 模型缓存版本统一为 `glm5_kpool`

## 5. 默认 graph 配置

启动脚本默认：

```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
--kernel-config '{"enable_jit_warmup": false}'
```

本次实测捕获结果：

- `FULL_DECODE_ONLY` 捕获 35 张 decode graph，最大 capture size 256。
- Graph 内存估算约 0.10 GiB，实际 graph pool 约 0.05 GiB/rank。
- 预填充和 chunked prefill 仍走 eager；当前 DLC 构建将 `torch.compile` 替换为透传，piecewise graph 没有可用子图。
- 当前环境自动回落到 V1 model runner，因为 V2 runner 要求 Triton，而本机没有可用的 Triton driver；本次 graph 修复针对实际使用的 V1 runner。

## 6. 验证结果

### 6.1 静态与单元测试

- `uvx ruff format`：本次变更文件统一格式化，42 个文件无需再改。
- `python -m compileall`：通过。
- `git diff --check`：通过。
- touched-file 全量 `ruff check`：通过（含 E402/E501/SIM 等规则，必要的延迟 import 已显式标注）。
- vllm-cl 完整定向回归：`161 passed, 10 subtests passed`。
- vllm 新增 graph padding 用例：独立等价脚本通过；整份上游测试文件受当前 PyTorch/Transformers conftest 兼容问题阻塞，不是本改动失败。

### 6.2 发布 smoke

配置：TP=8、`max_model_len=4096`、`max_num_batched_tokens=4096`、`max_num_seqs=64`、`gpu_memory_utilization=0.80`、`block_size=256`、graph 默认开启。

| 场景 | 结果 |
|---|---|
| 健康检查 | HTTP 200 |
| 单请求 20 prompt / 32 output | 2.107 s，HTTP 200 |
| 8 并发，每请求 20 / 32 | wall 3.751 s，256 output tokens，约 68.24 tok/s |
| 长上下文 3610 prompt / 8 output | 72.419 s，HTTP 200 |

服务冷启动约 3 分钟，其中模型权重加载约 110 s，KV/profile/graph 初始化约 33 s。以上结果在 `Driver Version_v26.0.1`、`Api_Version_26`、`DLC_SKIP_VERSION_CHECK=0` 下复测通过，日志中无版本错误、Traceback 或 NaN。

### 6.3 历史 graph/eager 性能参考

| 场景 | 并发 | Graph gap p50 | Eager gap p50 | Graph tok/s | Eager tok/s |
|---|---:|---:|---:|---:|---:|
| 49 prompt + 96 out | 1 | 67.64 ms | 110.27 ms | 14.11 | 8.83 |
| 49 prompt + 96 out | 8 | 103.49 ms | 112.86 ms | 65.38 | 59.95 |
| 49 prompt + 64 out | 32 | 222.04 ms | 222.88 ms | 100.08 | 100.55 |
| 49 prompt + 64 out | 64 | 277.89 ms | 263.59 ms | 92.10 | 99.12 |
| 860 prompt + 64 out | 1 | 73.34 ms | 96.03 ms | 11.27 | 8.99 |
| 860 prompt + 64 out | 8 | 148.58 ms | 144.84 ms | 30.46 | 31.22 |
| 1217 prompt + 8 out | 1 | TTFT 2.502 s | TTFT 2.498 s | — | — |

64 并发时 graph 低于 eager，主要来自请求形状落入 40-token bucket，不是结果正确性问题。

## 7. 合入主分支注意事项

1. 不要把整个 release 分支直接 merge 到 main。vllm 的自研改动需按目标分支 rebase/cherry-pick，vllm-cl 建议按三个提交线性应用，避免带入无关历史。
2. `vllm` 的改动集中在 `vllm/models/glm5next/common/kda.py`、`vllm/v1/attention/backends/gdn_attn.py` 及其测试；合并时优先保留目标分支的 attention metadata 结构，只移植 padding 边界逻辑。
3. `vllm-cl` 的 GLM 功能依赖 `vllm` 的 KDA `cu_seqlens` 与 GDN padding 修正；两个仓库应作为一组验证和发布。
4. 生产启动脚本和 compat stub 属于交付资产，放在 `work-docs`，不要复制回 vllm-cl 仓库。
5. 正式环境使用 `Driver Version_v26.0.1`（`Api_Version_26`）并通过严格版本检查；launcher 默认 `DLC_SKIP_VERSION_CHECK=0`，不要在生产环境关闭门禁。
6. 合并前重新跑 `compileall`、定向 pytest 和至少一次 graph smoke；若硬件卡死，先停服务，再执行 `cltechpd_clnt -s --full`。

## 8. 已知限制

- Graph 只覆盖 uniform decode，prefill 仍未 graph 化。
- graph capture size 最大到 256；不同并发和 token bucket 会影响命中率。
- 长上下文 prefill 仍是最主要的端到端耗时来源；当前发布不提供 native 慢路径门禁，07/09 融合算子统一用于 prefill 与 decode。
- AWQ MoE CPU fallback 仍需要主机侧布局逆变换和反量化辅助函数，这是生产必需代码，不是调试探针。
- 当前 V1 runner 是实际生产路径；V2 runner 在具备 Triton 的环境需要单独验证 graph、metadata 和 KPool 行为。
