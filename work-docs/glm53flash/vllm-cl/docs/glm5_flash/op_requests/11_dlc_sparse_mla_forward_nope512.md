<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22）：需修复已有算子问题（含 topk/分派小幅扩展） -->
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：已有算子但该配置不可用（建议改为「修复算子功能或精度」+ 扩展）
     证据：实测 512-NoPE（rope=0）输出 NaN 且复测卡死（kernel custom_sparse_attention_kv_cache_bf16_h4，head_dim=512/topk=2048）；576 布局 normal（err 3.91e-03）；topk 仅 256/512/1024/1536/2048，GLM5 需 2051
     建议：改为修复 + 扩展 dlc_sparse_attention_kv_cache：修复 rope=0 的 512 路径，并支持 topk 2051（可对齐 2176） -->

# [Operator Request] dlc_sparse_mla_forward_nope512（512 维 NoPE 稀疏 MLA 前向）

## 需求类型
修复算子功能或精度

**需求分类：需修复已有算子问题（含 topk/分派小幅扩展）**
- 现有算子：`torch.dlc_sparse_attention_kv_cache(query, kv_cache, topk_indices, scale, kv_lora_rank, return_softmax_lse=False, use_dual_xys_for_small_batch=True)`。
- 缺陷复现（dlc:0，`Q=2 / H=4`）：
  - `latent=512, rope=64, topk=2048` → `max_abs_err=3.91e-03`（正常，576 布局可用）；
  - `latent=512, rope=0, topk=2048`（**512 NoPE**）→ 输出 **NaN**，复测时 **kernel 卡死**；
  - 卡死时的 dump：`custom_sparse_attention_kv_cache_bf16_h4`，`head_dim=512`、`kv_cache=[512,256]`、`topk=[2048,2]`，scalar `0.0441942 / 512 / 0 / 1`。
  - 注意这是**数值崩/挂而不是参数拒绝**：`FusedOps.cpp:1255-1409` 只校验 `kv_cache.size(1) == head_dim`、`topk ∈ {256,512,1024,1536,2048}`、head 数 ∈ {2,4,8,64}，**没有对 `head_dim`/`kv_lora_rank` 做「必须 576」的显式检查**；512 布局因此会通过校验、进入按 576 布局编译的 kernel 后出错（这也是它表现为 NaN/卡死而非报错的原因）。
- 能力缺口（源码 `FusedOps.cpp: dlc_sparse_attention_kv_cache`）：topk 分派只覆盖 `256/512/1024/1536/2048`，GLM5 需要 **2051**（2048 个池 + 3 个 tail token；缓冲可对齐 2176）；head 数分派覆盖 `2/4/8/64`（GLM5 indexer 为 64 head，TP8 后 8，可用）。
- 期望修复后行为：① `rope=0`（512 NoPE）与 576 布局同样可用、数值对齐 fp32 reference；② topk 宽度接受 2051/2176（含 `-1` 无效槽）；③ 参数/布局不匹配时应报错返回而不是卡死。
- 插件侧早有同一结论的记录（本条目就是它的延伸）：`vllm_cl/attention/sparse_mla.py:552-557` 注明
  “The DLC sparse kernel currently supports the 576-wide latent+RoPE layout; use the reference PyTorch composition for the 512-wide path until a dedicated kernel is available”，
  并在 `forward_mqa` 里用 `kv_cache.shape[-1] == self.kv_lora_rank` 判断后回退到 `_forward_native_nope`（`sparse_mla.py:521/555`）；
  `SparseMLABackend.get_supported_head_sizes()`（`sparse_mla.py:169`）返回 `[512, 576]`，即 512 只是「声明支持、实现回退」。
- 复现脚本：`probes/probe_sparse_attention_nope512.py`（**会触发卡死，请单独运行并准备 Ctrl-C**）。
- 本次调整（2026-09-22）：需求类型「新增算子」→「修复算子功能或精度」；优先级 P0 不变；算子名称改为修复 + 小幅扩展 `torch.dlc_sparse_attention_kv_cache`（原提名的 `dlc_sparse_mla_forward_nope512` 不新增）。

## 优先级
P0

## 目标版本与期望时间
GLM-5.3-Flash 适配 Phase 2（性能阶段）。建议在长上下文（≥32K prefill）性能验收前提供，具体里程碑待算子组排期确认。

## 算子名称
修复 + 小幅扩展 `torch.dlc_sparse_attention_kv_cache`（512-NoPE 可用性 + topk 2051 分派；原提名 `dlc_sparse_mla_forward_nope512` 不新增）

## 背景与功能说明
使用场景：GLM-5.3-Flash 的 11 层 KPool-DSA 使用 `qk_rope_head_dim=0` 的 NoPE 稀疏 MLA，cache 中只保存 512 维 latent（`kv_lora_rank=512`），没有 64 维 RoPE 分量。
当前问题：DLC 现有稀疏注意力算子按 576 维（512 latent + 64 RoPE）实现，512 维 cache 无法直接使用。插件因此走了纯 PyTorch 回退 `_forward_native_nope`（`vllm_cl/attention/sparse_mla.py:569`）：手动 gather 选中槽位 → FP32 打分 → masked softmax → 加权聚合，并按 token/key 分块控制临时张量。这在长上下文下是主要的 decode/prefill 瓶颈（每次 gather `[Q, topk, 512]` 的 BF16 数据并做 FP32 softmax）。
目标功能：为稀疏 MLA 增加 512 维 NoPE 变体，输入吸收后的 query（`[Q, H_local, 512]`）、分页 latent cache、topk token 索引（宽度 2051 或对齐 2176，含 `-1` padding），输出稀疏聚合结果与 LSE。
非目标：KPool 索引的生成（07~10 号）；MLA 的 up-projection 与 `o_proj`（仍由现有 Linear 承担）。

## 期望 API 与函数签名
> **本条是「修复 + 小幅扩展已有算子」**：算子名保持 `torch.dlc_sparse_attention_kv_cache`，不新增独立算子；
> 修复点是让 `kv_lora_rank=512`（rope=0）的布局与 576 布局同样可用，扩展点是 topk 宽度接受 2051/2176、
> 并在 `head_dim`/`kv_lora_rank` 不受支持时报错返回（而不是卡死）。下面的签名是「修复后应支持的调用形态」。

```python
def dlc_sparse_attention_kv_cache(  # 现有算子，修复 512 路径 + topk 扩展
    q: torch.Tensor,               # [Q, H, 512] bf16, 吸收后 query（无 RoPE 分量）
    kv_cache: torch.Tensor,        # [num_blocks, 256, 512] bf16 或 [num_slots, 512]
    topk_indices: torch.Tensor,    # [Q, 2051] int32, -1 表示无效（缓冲可对齐 2176）
    softmax_scale: float,          # GLM-5.3-Flash: 1/sqrt(256)（沿用现有 512 路径的 scale）
    kv_lora_rank: int,             # 512
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # 返回 out [Q, H, 512] bf16（可选 lse [Q, H] fp32）
```

## 参数与返回值
输入：
- `q`：`[Q, H, 512]`，BF16，吸收后 query；H 为 TP 分片后的 head 数（64/TP）。
- `kv_cache`：`[num_blocks, 256, 512]` 或扁平的 `[num_slots, 512]`，BF16；GLM-5.3-Flash 无 RoPE 分量，因此尾维严格 512。
- `topk_indices`：`[Q, 2051]`，int32，取自 10 号算子的输出；`-1` 为无效槽。
- `softmax_scale`：float，GLM-5.3-Flash 沿用 `1/sqrt(256)`（**注意不是 `1/sqrt(512)`**，与现有 512 路径保持一致，请勿改写）。

属性：
- `kv_lora_rank`：512，用于校验 cache 尾维。
- `return_lse`：decode 需要返回 LSE 供合并使用。

返回值：
- `out`：`[Q, H, 512]`，BF16。
- `lse`：`[Q, H]`，FP32（`return_lse=True` 时）。

边界：`topk` 全为 `-1` 时输出为 0、LSE 为 `-inf`；`Q == 0` 返回空。

## 功能语义
对每个 query token 与 head：

```text
scores[i] = dot(q[n,h,:], cache[topk_ids[i], :]) * softmax_scale
scores[i] = -inf                     if topk_ids[i] == -1
p         = softmax(scores)          # FP32
out[n,h]  = sum_i p[i] * cache[topk_ids[i], :]
lse[n,h]  = logsumexp(scores)        # FP32
```

broadcasting / padding：`-1` 槽必须严格不参与 softmax 与聚合（贡献 0 概率）。
in-place / alias：不修改输入与 cache。
空 / 零长度：`Q == 0` 或 `topk == 0` 返回零输出。
异常：`kv_cache.shape[-1] != 512`、query 尾维不为 512（即出现 RoPE 分量）、`topk` 列宽不足报错。
确定性：无随机；softmax 与聚合 FP32。
数值稳定性：softmax 必须在 FP32 下做 max-subtraction；LSE 需与现有 576 路径口径一致。

## 支持范围与约束
支持的 dtype：`q`/`kv_cache` BF16，输出 BF16，LSE FP32；内部 FP32。
支持的 shape：`Q` 动态（prefill 到 32K+，decode 到 256+）；`H = 64/TP`（TP8 → 8）；latent 512 固定；`block_size=256`；`topk` 宽度 2051（或 2176 对齐）。
数值范围：latent FP32 聚合，BF16 存储。
layout：cache 支持 `[blocks, block_size, latent]` 与扁平 `[slots, latent]` 两种；block table 转换由调用方完成或按现有约定。
device：DLC 单卡。
执行模式：decode 主路径需支持 CUDA graph。
其他限制：
1. 现有 576 路径的 `concat_and_cache_mla` 与 sparse kernel 都硬编码 576，需要在同一需求中增加 512 分支（写侧见 12 号）。
2. 需要确认 kernel 是否接受 2051/2176 宽度与 `-1` padding；若不接受，请明确要求的对齐与哨兵约定。
3. DCP 索引映射（`convert_req_index_to_global_index`）在 512 路径下不能复用现有实现（native 已显式抛错），需一起确认。

## 精度要求
Golden / reference：`_forward_native_nope`（`vllm_cl/attention/sparse_mla.py:569`），FP32 softmax + FP32 聚合。
输入分布：`q/cache ~ N(0,1)`，`topk_indices` 覆盖全有效 / 部分 `-1` / 全 `-1`，含重复索引与越界索引。
样本数量：≥ 200 组，另加长上下文（topk 含量高）样本。
比较 dtype：BF16 输出对 FP32 reference；LSE 按 FP32 比较。
atol / rtol：`out atol=2e-2, rtol=2e-2`；`lse atol=1e-3, rtol=1e-3`。
必须逐元素一致：`-1` 槽不得改变结果；全 `-1` 时输出严格为 0、LSE 为 `-inf`。
NaN / Inf / 极值：score 极小时 softmax 不产生 NaN。
累加精度：score、softmax、聚合全程 FP32。
**scale 必须沿用 `1/sqrt(256)`**：这是 512 路径既有的约定，改动会造成与 reference 不一致。

## 测试范围
- [ ] 基本功能与 reference 对齐
- [ ] 多 dtype 精度
- [ ] 静态 shape 与动态 shape
- [ ] 最小值、最大值、空 tensor 等边界
- [ ] 非 contiguous、特殊 stride 或 layout
- [ ] 非法参数和异常处理
- [ ] 单卡
- [ ] 多卡/分布式
- [ ] 长时间稳定性
- [ ] 性能基准与性能回归

具体测试用例、shape、数据范围和预期结果：
1. 功能：Q ∈ {1, 64, 4096}，decode / prefill 两种形态。
2. topk：全有效、部分 `-1`、全 `-1`；宽度 2051 与 2176。
3. cache layout：`[blocks,256,512]` 与 `[slots,512]`。
4. `return_lse=True/False`；LSE 与 reference 一致。
5. 长上下文：Q=4096、topk=2048 时的耗时、峰值显存（对比 native 回退，目标 ≥ 5× 且显存显著下降）。
6. 边界 / 异常 / CUDA graph。
7. 交叉验收：与 12 号写入算子端到端联调（写入后立即读出，结果自洽）。

## 验证环境与依赖版本
DLC 硬件：8 卡，设备名 `/dev/cltech0-7`（TP=8，无 PP）
vLLM commit：`f5361fb09f`
vLLM DLC（vllm-cl）commit：`a14fc5b`
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：`VLLM_USE_DLC_COL_MAJOR_MATMUL=1`
模型：GLM-5.3-Flash-Mix（AWQ 4-bit，`linear_num_heads=64`、`linear_head_dim=128`、`short_conv_kernel_size=4`、`gate_lower_bound=-5.0`）

## 验收标准
- [ ] API、参数校验和错误行为符合约定
- [ ] 规定范围内的功能和精度测试通过
- [ ] 需求中明确列出的特殊运行场景通过
- [ ] 文档、测试和示例已提交
- [ ] 接入 vllm-cl 后，GLM-5.3-Flash 在 DLC 上 prefill/decode 数值与 PyTorch reference 一致

## Reference、日志与附件
Reference 实现：`vllm_cl/attention/sparse_mla.py:569`（`_forward_native_nope`），入口判断见 `:512`（`_forward_bf16_kv` 中 `native_nope = kv_cache.shape[-1] == self.kv_lora_rank`）。
现有 576 路径：`vllm_cl/_custom_ops.py` 的 `sparse_attention`（`torch.dlc_sparse_attention_kv_cache`）。
backend：`vllm_cl/attention/sparse_mla.py:140`（`get_kv_cache_shape`）、`:152`（`get_kv_cache_block_dim`）。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md` §6.6（NoPE 稀疏 MLA 的 4 项待验证点）。
