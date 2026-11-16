# Hy4-preview-Mix DLC 适配方案

- 日期：2026-11-13
- 模型：`/data/models/Hy4-preview-Mix`
- 目标基线：vLLM `upstream-v0.30.1rc0` + vllm-cl `v0.30.1rc0`
- 适配分支：`feat/hy4-preview-adapt`（vllm、vllm-cl 同名）

## 1. 结论先行

1. **模型主干不需要重写。** 当前 vLLM 基线已经包含 `HYV4ForCausalLM`、`HYV4MTP`、
   `HYV4Config`、tokenizer/tool/reasoning parser，以及 NVIDIA/ROCm 平台实现。
   Hy4 的 DLC 工作本质上是平台移植：保留模型层、替换注意力/索引器/量化 MoE 的
   DLC 后端，并补齐 DLC 专用融合算子。
2. **P0 算子缺口集中在 iHC 和 attention sink。** iHC 的 eager 路径每 token 约
   `156 × (20 + 5) ≈ 3900` 次 kernel launch；每层 MLA 的 learnable sink 当前
   没有 DLC 路径。两者都会直接影响生产可用性。
3. **MoE 不需要新 kernel，但必须修两个接入问题。** 现有 W4A16 AWQ MoE kernel
   已支持 group64、256 experts、FP32 router logits 和动态 `swiglu_limit`；
   vllm-cl 当前却把 `swiglu_limit` 写死为 `7.0`，并把 FP32 router logits 强制
   转成 BF16。Hy4 的配置是 `swiglu_limit=10.0`，且 router 权重/输出为 FP32。
4. **索引器首期走 BF16 cache。** DLC 已有 BF16 indexer cache + paged MQA + top-k
   路径；Hy4 上游使用 FP8/UE8M0 indexer cache。首期复用 BF16 路径，先保证打通，
   再根据 top-k 一致性和性能决定是否补 FP8 路径。
5. **内存是硬门槛。** 权重约 `427.27 GiB`，TP8 每卡下限约 `53.4 GiB`，而单卡
   只有 `64 GiB` HBM。初始必须使用约 `0.95` 以上显存利用率，并限制并发/上下文；
   1M 上下文是模型能力，不是当前 TP8 BF16 部署能力。

## 2. 仓库基线与环境状态

2026-11-13 清理并更新后的状态：

| 仓库 | 分支 | HEAD | 说明 |
|---|---|---|---|
| LLVM | `main` | `66c6fd00f204` | 最新 main |
| CLsim | `main` | `bedfd94` | 最新 main |
| DLCSynapse | `main` | `bbbe1337` | 最新 main |
| DLC_Custom_Kernel | `main` | `9bcbb426e` | 最新 main |
| DLC_CL | `main` | `ff20b34` | 最新 main |
| pytorch | `main` | `d0221a52cc` | 最新 main |
| vllm | `feat/hy4-preview-adapt` | `108b49ea10` | 基于 `upstream-v0.30.1rc0` |
| vllm-cl | `feat/hy4-preview-adapt` | `89731b3` | 基于最新 `origin/v0.30.1rc0` |

其他环境事实：

- `/dev/cltech0..7` 可见，设备名 `TYD`，单卡 `64 GiB` HBM。
- 驱动版本为 `Driver Version_v26.1.7`，与 AGENTS 中 GLM 发布基线记录的
  `26.0.1` 不同，需要在首次跑通时重新确认兼容性。
- CRT 四个 hex 均为 `0TYD`，`dlc_crt.rtabi=3.2.0`。
- `libcustom_dlc_perf_lib.so` 约 `1.6 GiB`（2026-11-13 08:39）。
- 当前 editable 安装仍指向旧提交：vLLM 为 `13b663fd0`，vllm-cl 为 `10a2fbbbe`。
  切换分支后 Python 源码会走新分支，但编译扩展可能仍是旧产物；正式验证前必须
  重新构建并安装 vLLM/vllm-cl（必要时同步重建 custom kernel）。
- 2026-11-13 预检已确认该问题：导入新 `vllm.transformers_utils.config` 时，
  旧 `vllm_cl_C*.so` 缺少 `dispatch_key`，报
  `ImportError: cannot import name 'dispatch_key' from 'vllm_cl.vllm_cl_C'`。
  在重建 vllm-cl C++ 扩展前，不进行任何依赖 vLLM import 的模型测试。
- `vllm-cl/v0.30.1rc0` 曾被远端强推。旧本地基线保留在
  `backup/v0.30.1rc0-local-20261113`，当前 `v0.30.1rc0` 已对齐
  `origin/v0.30.1rc0`。

## 3. 模型结构与权重

### 3.1 关键配置

| 配置 | 值 |
|---|---:|
| `model_type` / `architectures` | `hy_v4` / `HYV4ForCausalLM` |
| hidden size | 6144 |
| decoder layers | 78 |
| MLA heads | 64 |
| q / kv LoRA rank | 2048 / 512 |
| qk nope / rope / v head dim | 192 / 64 / 256 |
| dense intermediate | 18432 |
| routed experts | 256 |
| experts per token | 8 |
| routed expert intermediate | 2048 |
| shared experts | 1 |
| indexer heads / dim / top-k | 32 / 128 / 2048 |
| indexer full / shared layers | 21 / 57 |
| dense / sparse MLP layers | 1 / 77 |
| iHC channels / magnitude / eps | 4 / 2.0 / 1e-6 |
| gated MLA | elementwise |
| learnable sink | true，64 heads，FP32 |
| routed SwiGLU clamp | 10.0 |
| MTP layers | 1 |
| 最大位置 | 1,048,576 |
| 量化 | AWQ 4-bit，group 64，带 zero point，`awq_marlin` |

### 3.2 权重体量

索引元数据总大小 `458,799,808,492 B`，按实际 tensor header 统计约
`427.27 GiB`。79 个 shard 均已存在，索引无缺失。

| 类别 | 体量 |
|---|---:|
| routed experts | 373.57 GiB |
| attention + indexer | 38.97 GiB |
| MTP | 5.58 GiB |
| shared experts | 5.41 GiB |
| embedding / lm_head / final norm | 2.77 GiB |
| layer0 dense MLP | 0.63 GiB |
| iHC 参数 | 0.11 GiB |
| 其他 | 0.23 GiB |

TP8 静态权重下限约 `53.4 GiB/卡`，未计入 MTP/参数副本、workspace、graph pool、
KV cache 和碎片。按 `64 GiB × 0.95 = 60.8 GiB` 计算，理论剩余约 `7.4 GiB`；
实际可用空间会更小。

### 3.3 关键张量形状

| 张量 | 形状 |
|---|---|
| `q_a_proj.weight` | `[2048, 6144]` BF16 |
| `q_b_proj.weight` | `[16384, 2048]` BF16 |
| `kv_a_proj_with_mqa.weight` | `[576, 6144]` BF16 |
| `kv_b_proj.weight` | `[28672, 512]` BF16 |
| `o_proj.weight` | `[6144, 16384]` BF16 |
| `linear_gate.weight` | `[16384, 6144]` BF16 |
| `learnable_sink_param` | `[64]` FP32 |
| indexer `wk` / `wq_b` / `weights_proj` | `[128,6144]` / `[4096,2048]` / `[32,6144]` |
| 单个 expert `gate_proj.qweight` | `[6144, 256]` I32，group64 打包 |
| 单个 expert `gate_proj.scales` | `[96, 2048]` BF16 |
| `hc_pre.hc_fn` | `[8, 24576]` FP32 |
| `hc_head.hc_head_fn` | `[4, 24576]` FP32 |

## 4. 现有上游 Hy4 实现

关键入口：

- `vllm/vllm/models/hy_v4/__init__.py`
- `vllm/vllm/models/hy_v4/nvidia/model.py`
- `vllm/vllm/models/hy_v4/nvidia/attention.py`
- `vllm/vllm/models/hy_v4/nvidia/hc.py`
- `vllm/vllm/models/hy_v4/nvidia/moe.py`
- `vllm/vllm/models/hy_v4/nvidia/mtp.py`
- `vllm/vllm/models/hy_v4/nvidia/flashmla_sparse.py`
- `vllm/vllm/transformers_utils/configs/hy_v4.py`

上游已经实现：

- iHC pre/post/head 的 eager 语义。
- MLA + DSA top-k + gated MLA + learnable sink。
- sigmoid routed MoE、shared expert、W4A16 AWQ。
- MTP draft layer、top-k buffer 共享、spec decode。
- HYV4 parser 和模型注册。

当前入口对非 ROCm 平台默认选择 `nvidia/`。DLC 不是 CUDA，但会落到该目录，
因此适配方式是：

1. vllm-cl 注册/补丁 DLC 平台实现，不改上游公共模型语义。
2. 复用上游 `HYV4DecoderLayer`、`HYV4Model`、`HYV4MoEFused`、`HYV4MTP` 的结构。
3. 对 iHC、indexer、sparse MLA sink、AWQ MoE 做 DLC 专用替换。

## 5. 差距矩阵

| 模块 | 上游 Hy4 | DLC 当前 | 适配动作 | 优先级 |
|---|---|---|---|---|
| 模型/配置/MTP 注册 | 已支持 | 基线源码可用 | vllm-cl 增加 `hy_v4` 平台补丁 | P0 |
| iHC pre/post/head | Triton/HPC + eager fallback | 可用 eager，但 launch 数过高 | 新增 3 个融合算子 | P0/P1 |
| sparse MLA | FlashMLA sparse + sink | DLC sparse attention 支持 576/top-k 2048，但无 sink | 新增 sink epilogue，后续可融合进 attention | P0 |
| gated MLA | HPC 融合或 eager | eager linear+sigmoid+mul | 首期 eager；生产前加融合 epilogue | P1 |
| indexer | FP8/UE8M0 cache + fused q/k | BF16 cache + paged MQA 路径 | 新增 DLC HYV4 Indexer 包装，首期 BF16 | P0 |
| shared/full indexer | 复用最近 full 层 top-k | `DLCSparseAttnIndexer` 已有 buffer 机制 | 保持 `indexer_types` 语义，验证逐层复用 | P0 |
| AWQ MoE | FP32 router + `swiglu_limit=10` | kernel 支持，但 Python 写死 7.0 且转 BF16 | 修 vllm-cl 接入并补 256-expert 回归 | P0 |
| router gate | FP32 6144→256 | `DLCGateLinear` 可走 FP32 | 验证 shape/精度，不做新 kernel | P0 |
| MTP | 独立 top-k buffer + graph | GLM MTP 路径可复用 | 移植到 Hy4，验证 verify/graph | P1 |
| KV cache | MLA 576 | DLC MLA cache 路径已有 | 计算容量，先 4K/8K | P0 |
| graph | FULL_DECODE_ONLY | DLC graph 已有 | 先 eager，再 decode graph | P1 |
| DCP/CP | NVIDIA 支持 | DLC sparse/DCP 部分支持 | 首期 DCP=1，不作为 bring-up 阻塞 | P2 |

## 6. 算子需求清单

详细需求位于 `work-docs/hy4-preview/operator_requests/`。

| 优先级 | 算子 | 类型 | 作用 |
|---|---|---|---|
| P0 | `torch.dlc_ihc_pre` | 新增 | iHC 通道降维 + pre/post gate |
| P0 | `torch.dlc_ihc_post` | 新增 | post gate 散射 + 多通道 residual |
| P1 | `torch.dlc_ihc_head` | 新增 | 最终 iHC 通道合并 |
| P0 | `torch.dlc_sparse_mla_apply_sink_` | 新增 | 将 learnable sink 应用到 sparse MLA 输出 |
| P1 | `torch.dlc_gated_mla_epilogue` | 新增 | gated MLA 的 gate GEMM + sigmoid + 乘 |
| P0 | `torch.dlc_vllm_fused_moe_w4a16_awq_col_major` | 扩展/修复 | FP32 router、`swiglu_limit=10`、256 experts 回归 |

不在首批新算子范围：

- 不新增 indexer BF16 cache 算子；复用现有
  `torch.dlc_indexer_k_cache_update_`、`cp_gather_indexer_k_cache`、
  `multi_query_attention`、`top_k_per_row_decode`。
- 不新增 AWQ 反量化算子；复用现有 W4A16 AWQ MoE。
- 不新增 MTP 专用算子；复用 GLM MTP/EAGLE 路径。
- iHC `post+next pre` 跨层融合是 P2 性能优化，不进入首批需求。

## 7. 分阶段实施计划

### Phase 0：基线可加载（1–2 天）

目标：

- 重建/安装 vLLM + vllm-cl 新基线，确认 editable 扩展与源码一致。
- 只做 config/tokenizer/weight-map 加载，不启动完整模型。
- 建立 Hy4-mini（小 hidden、少层、少 expert）CPU reference。

验收：

- 解决旧的 `vllm_cl_C*.so` 缺少 `dispatch_key` 的导入错误，确认扩展与
  `feat/hy4-preview-adapt` 源码一致。
- `AutoConfig` 解析为 `HYV4Config`，关键字段与 checkpoint 一致。
- 79 个 shard 索引无缺失；`lm_head`、expert、MTP、iHC 张量均能映射。
- DLC 设备、CRT、RTABI、custom kernel 版本检查通过。

### Phase 1：单层/小模型语义对齐（3–5 天）

目标：

- 实现 DLC HYV4 Indexer 包装（BF16 cache）。
- 接入 iHC 三个算子；未交付前允许 eager fallback，但只用于语义验证。
- 接入 sparse MLA sink epilogue。
- 修复 AWQ MoE 的 `swiglu_limit` 和 FP32 router。
- 用 Hy4-mini 对比 DLC 与 CPU reference 的 layer output/logits。

验收：

- iHC pre/post/head 单算子误差达标。
- sparse MLA + sink 与 F32 reference 对齐。
- 单层 MoE 路由 top-k 一致，专家输出误差达标。
- `swiglu_limit=10` 与 `=None`、`=7` 的结果有明确区分且符合 reference。

### Phase 2：TP8 全模型打通（3–5 天）

目标：

- 以 TP8 加载完整模型，先用 `max-model-len=4096`、`max-num-seqs=8`。
- 比较 TP8 no-EP 与 TP8+EP8 两种 MoE 布局。
- 验证 prefill、decode、chunked prefill、短请求输出。

建议初始配置：

```bash
export VLLM_PLUGINS=dlc,dlc_patches
export DLC_SKIP_VERSION_CHECK=0
export PYTORCH_DLC_ALLOC_CONF=expandable_segments:True

vllm serve /data/models/Hy4-preview-Mix \
  --served-model-name Hy4-preview-Mix \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --port 18005
```

MoE 布局先测：

- `TP8`：不启用 expert parallel，expert intermediate 每 rank 为 `2048/8=256`。
- `TP8 + EP8`：启用 expert parallel，并设置 DLC legacy `--moe-ep-size 8`，
  每 rank 32 个完整 expert。

验收：

- 服务启动、健康检查、单请求输出非 NaN。
- 同一 prompt 在 eager 下可重复。
- 记录权重加载时间、峰值 HBM、KV block 数。

### Phase 3：Graph、MTP 与并发（5–8 天）

目标：

- eager 稳定后启用 `FULL_DECODE_ONLY`。
- 接入 Hy4 MTP，验证 target/draft top-k buffer 共享。
- 测 c1/c2/c4/c8 并发和 graph bucket padding。

验收：

- graph 与 eager 同输入输出一致。
- MTP 不污染 target top-k buffer；draft step 0 的 indexer 行为正确。
- 无非预期 D2H sync、无图捕获失败。

### Phase 4：性能与长上下文（持续）

目标：

- 评估 iHC 融合、gated MLA epilogue、MoE 布局、indexer BF16/FP8 的成本。
- 逐步扩大 `max-model-len`，直到显存或性能门槛。

验收：

- 给出 4K/8K/16K/32K 的 TTFT、decode tok/s、峰值 HBM。
- 明确 1M 上下文是否必须引入 FP8 KV cache、DCP/CP 或多机。

## 8. 内存与上下文估算

每 token KV 粗算（BF16 MLA cache）：

- MLA：`78 × 576 × 2 = 89,856 B ≈ 87.75 KiB`
- BF16 indexer cache（21 个 full 层）：`21 × 128 × 2 = 5,376 B ≈ 5.25 KiB`
- 合计约 `93.0 KiB/token`

对应单序列：

| 上下文 | MLA + indexer cache |
|---:|---:|
| 4K | 约 372 MiB |
| 8K | 约 744 MiB |
| 32K | 约 2.91 GiB |
| 128K | 约 11.6 GiB |
| 1M | 约 93 GiB |

在剩余约 7 GiB、block size 256 时：

- 每个 KV block 约 `95,232 × 256 ≈ 23.25 MiB`。
- 理论上限约 300 个 block，即约 76K token 的聚合容量，尚未扣除 workspace、
  graph pool 和碎片。
- 4K/请求需要 16 个 block；初始 c8 是合理起点，c16 以上必须实测。

结论：

- `max-model-len=4096` 或 `8192` 可作为 bring-up 配置。
- 1M 上下文在当前 TP8 BF16 权重布局下不可部署；需要额外手段。

## 9. 验证矩阵

| 层级 | 用例 | 通过标准 |
|---|---|---|
| 算子 | iHC pre/post/head | 与 F32 reference 达标，T=0/1/边界 shape 通过 |
| 算子 | sparse MLA + sink | sink=-inf/0/大值、无效 top-k、全 mask 行通过 |
| 算子 | AWQ MoE | 256 experts、group64、F32 router、limit 10 通过 |
| 单层 | 1 dense + 1 MoE + 1 shared indexer + 1 full indexer | 与 CPU reference 对齐 |
| 小模型 | Hy4-mini 2 层 | logits/top-k 对齐，NaN 检查通过 |
| 全模型 | TP8 4K 单请求 | 输出稳定，无 NaN，显存可回收 |
| 全模型 | c1/c2/c4/c8 | 并发不串扰，KV block 正确 |
| Graph | eager vs FULL_DECODE_ONLY | 同输入逐 token 一致 |
| MTP | target + draft | top-k buffer 共享正确，spec verify 通过 |
| 长上下文 | 4K/8K/16K/32K | 记录 TTFT、decode、峰值 HBM |
| 回归 | GLM-5.3-Flash 冒烟 | 新分支不破坏既有模型 |

## 10. 主要风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 权重 427 GiB、单卡 64 GiB | 并发/上下文极低，OOM | 0.95 利用率、c8 起步、测 EP/TP 两种布局 |
| iHC eager 路径 launch 过多 | decode 无法接受 | P0 交付 pre/post，P1 交付 head |
| sink 未生效 | 输出系统性偏差，难定位 | 首期强制 sink epilogue，禁止静默禁用 |
| indexer BF16 vs 上游 FP8 | top-k 选点可能不同 | 与 reference 对比 top-k overlap；必要时补 FP8 cache 路径 |
| router FP32→BF16 | 专家路由翻转 | 首期禁止 cast，kernel 已支持 FP32 |
| `swiglu_limit` 写死 7 | 专家输出偏差 | 从 `layer.swiglu_limit` 读取，固定回归 10.0 |
| MTP top-k buffer 共享 | draft 读取错误索引 | 复用 GLM MTP 的 set/compact 逻辑并加断言 |
| 新基线扩展未重建 | 源码/二进制不一致 | Phase 0 强制重建并记录 commit |
| 驱动 26.1.7 非 GLM 验证版本 | 兼容性未知 | 首次跑通记录驱动/RTABI/kernel 版本，异常时回退 |
| 1M 上下文预期 | 直接开长上下文会 OOM | 明确分阶段扩容，先 4K/8K |

## 11. 立即行动项

1. 重建并安装 vLLM、vllm-cl 新基线，记录实际 import 版本和 `.so` 时间戳。
2. 在 vllm-cl 增加 `hy_v4` 平台模块：
   - DLC HYV4 Indexer（BF16 cache、HYV4 的 `[nope, rope]` 布局）。
   - `supports_sink=True` 的 DLC sparse MLA 包装；`PyTorchMLASparseImpl`
     接收 `sinks`，并用 `lse_dtype=torch.float32` 请求原始 LSE 后调用 sink epilogue。
   - iHC 算子替换入口，保留 eager fallback。
   - AWQ MoE 的 FP32 router 和动态 `swiglu_limit`。
3. 建立 Hy4-mini reference 与算子单测，先验证 iHC、sink、MoE。
4. 并行推进算子组 P0 需求：`ihc_pre`、`ihc_post`、`sparse_mla_apply_sink_`。
5. TP8 首次加载只做 4K/单请求，记录峰值 HBM 后再决定 MoE 布局和并发。

## 12. 参考

- `vllm/vllm/models/hy_v4/nvidia/model.py`
- `vllm/vllm/models/hy_v4/nvidia/attention.py`
- `vllm/vllm/models/hy_v4/nvidia/hc.py`
- `vllm/vllm/models/hy_v4/nvidia/moe.py`
- `vllm/vllm/models/hy_v4/nvidia/mtp.py`
- `vllm-cl/vllm_cl/ops/awq_marlin.py`
- `vllm-cl/vllm_cl/attention/sparse_mla.py`
- `vllm-cl/vllm_cl/ops/sparse_attn_indexer.py`
- `DLC_Custom_Kernel/dlc_kernels/vllm/mhc_pre.cpp`
- `DLC_Custom_Kernel/dlc_kernels/vllm/mhc_post.cpp`
- `DLC_Custom_Kernel/dlc_kernels/moe/router.hpp`
- `pytorch/aten/src/ATen/native/dlc/FusedOps.cpp`

