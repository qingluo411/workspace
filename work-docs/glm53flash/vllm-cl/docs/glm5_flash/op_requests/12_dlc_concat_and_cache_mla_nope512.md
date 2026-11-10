<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22）：可复用已有算子（本条撤回，不提交） -->
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：已被 dlc_vllm_concat_and_cache_mla_ 完全覆盖（建议撤回，不要提单）
     证据：实测 kpe=0（512 NoPE）与 kpe=64（576）均 bit-exact（latent_err=0.000e+00 kpe_err=0.000e+00）；要求 slot_mapping int64、kv_cache contiguous、block_size 为 2 的幂、kvc_dim 为 256 倍数
     建议：撤回本需求；插件按上述约束直接调用即可（当前走 native 只是因为判断了 k_pe.shape[-1]==0） -->

# [Operator Request] dlc_concat_and_cache_mla_nope512（512 维 NoPE MLA cache 写入）

## 需求类型
其他

**需求分类：可复用已有算子（本条撤回，不提交）**
- 现有算子：`torch.dlc_vllm_concat_and_cache_mla_(kv_cache, kvc, kpe, slot_mapping)`。源码要求（`FusedOps.cpp: dlc_vllm_concat_and_cache_mla_impl`）：`slot_mapping` 必须 **int64**、`kv_cache` 必须 contiguous、`block_size` 为 2 的幂、bf16 下 `kvc_dim` 为 256 的倍数；`kpe_dim < 256`，因此 **kpe 宽度为 0 是合法的**。
- 实测（dlc:0，block_size=256）：
  - `kpe=64`（576 布局）→ `latent_err=0.000e+00  kpe_err=0.000e+00`；
  - `kpe=0`（512 NoPE）→ `latent_err=0.000e+00`。
  即 512 维 NoPE 场景已被完全覆盖（bit-exact），**无需扩展**。
- 插件现状：`vllm_cl/attention/sparse_mla.py:60` 因 `k_pe.shape[-1] == 0` 的判断走了 native；按上述约束改为直接调用即可，本 issue 撤回。
- 注意插件里有**两处**关于「只支持 576」的旧注释，所指的不是同一个算子，别混淆：
  1. `sparse_mla.py:68-70`（cache **写入** `concat_and_cache_mla_dlc`）说 “The fused DLC writer currently only accepts the 576-wide latent+RoPE layout” —— **这条已过时**：`dlc_vllm_concat_and_cache_mla_` 在 `kpe=0` 下实测 bit-exact（`kpe_dim < 256` 即合法，0 也合法），接入时应把该 early-return 去掉、直接走融合写入。
  2. `sparse_mla.py:552-557`（稀疏注意力**前向**）说 “The DLC sparse kernel currently supports the 576-wide latent+RoPE layout” —— **这条是准确的**，就是 11 号那条 P0 缺陷（512-NoPE 走 `_forward_native_nope` 回退）。
- 复现脚本：`probes/probe_concat_and_cache_mla_nope.py`。
- 本次调整（2026-09-22）：需求类型「扩展已有算子能力」→「其他（复用已有算子）」；优先级 P1→P2；算子名称改为直接调用 `torch.dlc_vllm_concat_and_cache_mla_`；**本条撤回，不提交**。

## 优先级
P2

## 目标版本与期望时间
本条不提交给算子组、不需要排期：插件侧在 Phase 2 内把 512-NoPE 写入切到现有 `torch.dlc_vllm_concat_and_cache_mla_`（算子组无需改动）。
## 算子名称
复用 `torch.dlc_vllm_concat_and_cache_mla_`（`kpe` 宽度为 0 已支持；原提名 `dlc_concat_and_cache_mla_nope512` 撤回）

## 背景与功能说明
使用场景：11 层 KPool-DSA 在 prefill/chunked prefill 阶段把 `kv_a_layernorm` 后的 512 维 latent 写入分页 MLA cache。GLM-5.3-Flash 是 NoPE（`qk_rope_head_dim=0`），没有 k_pe 分量。
当前问题：DLC 现有融合写入算子 `torch.dlc_vllm_concat_and_cache_mla_` 只接受 576 维（512 latent + 64 RoPE）布局。插件因此对 512 路径走普通索引赋值（`vllm_cl/attention/sparse_mla.py:60` 的 `concat_and_cache_mla_dlc` 分支）：

```python
kv_cache[valid_slots // block_size, valid_slots % block_size] = values[valid]
```

这是分散的 index-copy，prefill 大批量写入时带宽利用率低，且会打断 graph。
目标功能：让融合写入算子支持 `k_pe` 宽度为 0 的 512 维路径，直接按 slot_mapping 写入 latent。
非目标：读出侧（11 号）；KPool 池化（07 号）。

## 期望 API 与函数签名
> **本条已撤回（不提交）**：下面这段签名是初稿拟定的形式，仅作存档；实际动作是复用现有算子 `torch.dlc_vllm_concat_and_cache_mla_`
> （见「需求分类」段与「算子名称」），**不需要算子组提供这个新 API**。

```python
def concat_and_cache_mla_nope512(
    k_c_normed: torch.Tensor,   # [T, 512] bf16, 已过 kv_a_layernorm 的 latent
    kv_cache: torch.Tensor,     # [num_blocks, 256, 512] bf16, 原位写
    slot_mapping: torch.Tensor, # [T] int32, -1 表示跳过
    kv_cache_dtype: str = "auto",
) -> None:
    # 等价于现有 concat_and_cache_mla(k_c_normed, k_pe=None, ...)
```

## 参数与返回值
输入：
- `k_c_normed`：`[T, 512]`，BF16，latent。
- `kv_cache`：`[num_blocks, 256, 512]`，BF16，原位写。
- `slot_mapping`：`[T]`，int32，物理槽；`-1` 跳过（不出错）。
- `kv_cache_dtype`：与现有算子保持一致。

属性：
- 允许 `k_pe` 缺省或宽度为 0（建议实现为 `k_pe=None` 的可选参数，保持与 576 路径同一入口）。

返回值：无（原位写）。

边界：`T == 0` 无操作；`slot_mapping` 全为 `-1` 无操作。

## 功能语义
```text
for t in range(T):
    s = slot_mapping[t]
    if s < 0: continue
    kv_cache.flat[s] = k_c_normed[t]      # 尾维 512
```

in-place：只写 `kv_cache`。
空 / 零长度：`T == 0` 无操作。
异常：`kv_cache.shape[-1] != 512`、`k_c_normed.shape[-1] != 512`、`slot_mapping` 长度不符报错。
确定性：无随机。

## 支持范围与约束
支持的 dtype：BF16（`kv_cache_dtype` 与现有算子一致）。
支持的 shape：`T` 动态；latent 512 固定；block_size=256。
layout：`k_c_normed` 连续；`kv_cache` 支持 `[blocks, block_size, latent]`。
device：DLC 单卡。
执行模式：需支持 CUDA graph。
其他限制：与 576 路径共用一个入口时，必须保证 576 行为完全不变（回归测试覆盖）。

## 精度要求
Golden / reference：`concat_and_cache_mla_dlc` 的 512 分支（`vllm_cl/attention/sparse_mla.py:60`）。
输入分布：随机 latent；`slot_mapping` 覆盖连续 / 乱序 / 含 `-1` / 越界。
样本数量：≥ 200 组。
比较 dtype：cache 内容按 BF16 逐元素比较（只比较被写入的槽）。
atol / rtol：**要求 bit-exact**（纯搬运 + 拷贝）。
必须逐元素一致：`-1` 槽不得被修改（需预填哨兵值验证）；未命中的 cache 位置保持不变。
NaN / Inf / 极值：原样搬运。

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
1. 功能：T ∈ {1, 256, 4096}；slot_mapping 连续 / 乱序 / 含 -1。
2. 不覆盖：`-1` 槽与未选中槽必须保持原值（预填哨兵）。
3. 回归：576 路径行为不变（跑一次现有 576 用例）。
4. 边界 / 异常 / CUDA graph。
5. 性能：T=4096 下写入带宽；对比 native 索引赋值（目标 ≥ 2×）。

## 验证环境与依赖版本
DLC 硬件：8 卡，设备名 `/dev/cltech0-7`（TP=8，无 PP）
vLLM commit：`f5361fb09f`
vLLM DLC（vllm-cl）commit：`a14fc5b`
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：`VLLM_USE_DLC_COL_MAJOR_MATMUL=1`
模型：GLM-5.3-Flash-Mix（AWQ 4-bit，`linear_num_heads=64`、`linear_head_dim=128`、`short_conv_kernel_size=4`、`gate_lower_bound=-5.0`）

## 验收标准
- [ ] 插件侧把 `vllm_cl/attention/sparse_mla.py:60` 的 `k_pe.shape[-1] == 0` 提前返回改为直接调用融合写入，并同步修正 `:68` 的过时注释。
- [ ] 约束满足：`slot_mapping` int64、`kv_cache` contiguous、`block_size` 为 2 的幂、bf16 下 `kvc_dim % 256 == 0`、`kpe_dim < 256`（0 合法）。
- [ ] 512-NoPE 写入 bit-exact（实测 `latent_err=0.000e+00`），576 布局回归不变（`kpe_err=0.000e+00`）。
- [ ] 不修改现有算子：本条不产生算子组接口/行为变更。
## Reference、日志与附件
Reference 实现：`vllm_cl/attention/sparse_mla.py:60`（`concat_and_cache_mla_dlc` 的 512 分支）；576 路径为 `torch.dlc_vllm_concat_and_cache_mla_`。
调用点：`vllm_cl/attention/sparse_mla.py` 的 MLA 写入路径；`vllm/models/glm5next/nvidia/attention.py`。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md` §6.6 第 1 项（`concat_and_cache_mla_dlc` 是否允许空 `k_pe`）。
