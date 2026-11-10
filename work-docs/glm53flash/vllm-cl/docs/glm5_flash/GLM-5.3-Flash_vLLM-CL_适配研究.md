# GLM-5.3-Flash 适配 vLLM-CL 研究与实施建议

> 研究日期：2026-09-01  
> 目标：明确把 `zai-org/GLM-5.3-Flash` 接入 DLC 自研芯片 vLLM 插件所需的模型、缓存、算子、权重加载、多模态和验证改造。本文是适配设计文档，不代表已经在 DLC 真机完成实现或验证。

## 1. 结论摘要

GLM-5.3-Flash 不能按“复用 GLM-5.2/GLM-4 MoE，仅补一个模型注册”处理。它引入的是一套混合状态模型：34 层 KDA 线性注意力、11 层带 KPool 索引器的 NoPE 稀疏 MLA、45 层四路 mHC、42 层 MoE、一个原生多模态视觉塔，以及一层可选 MTP。

对当前 `vllm-cl`，适配工作可以分成四类：

1. **必须新增的执行路径**：KDA 的短卷积、prefill、decode 和状态更新；KPool 压缩、尾池缓存和池索引展开；NoPE 稀疏 MLA 的 512 维缓存路径。
2. **已有基础但需修正的路径**：稀疏 MLA、TopK、MoE、混合 Mamba/Attention 缓存、视觉卷积和 MTP 框架。
3. **可先用 PyTorch reference、后做融合的路径**：mHC、MTP 的 `eh_norm`、部分 KPool 变换。
4. **部署级阻塞项**：当前 `vllm-cl` 声明 `supports_fp8=False`，而默认开源 checkpoint 是 block-FP8；固定 `block_size=256` 又与上游 PR 的共享页 KDA/MLA 缓存布局存在尺寸冲突，必须走 DLC 专用的分池分配方案。

建议先完成 **BF16、文本、无 MTP、TP/EP、不启用 PP** 的正确性闭环，再依次增加多模态、FP8/AWQ、MTP、前缀缓存和长上下文性能。

## 2. 研究基线与证据

### 2.1 本地资料

- [GLM-5.3-Flash 官方模型配置](./02_GLM-5.3-Flash_config.json)
- [Z.ai 官方发布文章](./01_ZAI_GLM-5.3-Flash_Blog.html)
- [NVIDIA NeMo 架构说明](./03_NVIDIA_NeMo_GLM-5.3-Flash.md)
- [Raschka 架构笔记](./04_Raschka_GLM-5.3-Flash_Architecture_Notes.html)
- [Kimi K3 技术报告](./k3_tech_report.pdf)，重点参考 §2.1.1、§5.1、§5.4.1 和 §5.4.2。

### 2.2 代码基线

| 对象 | 本次检查版本 | 结论 |
|---|---:|---|
| 本地官方 vLLM | `45aed9b0cd`，2026-08-31 | `main` 中尚无 `vllm/models/glm5next/` |
| 上游 GLM 支持 | PR [vllm#53906](https://github.com/vllm-project/vllm/pull/53906)，head `f3ba9c0553` | 截至研究日仍为 open；包含模型、KDA、KPool、缓存管理、多模态和 MTP 改动 |
| 本地 vllm-cl | `252b5c8358`，2026-08-27 | 已适配 GLM4-MoE、GLM4.1V、DeepSeek 稀疏 MLA 等，但没有 GLM5Next/KDA/KPool/mHC 专用实现 |
| vllm-cl 对齐文件 | `v0.23.1rc0 / 71e2bbbc39` | 与本地官方 vLLM `main` 不同代；模型适配前要先完成源码对齐 |
| Transformers 参考实现 | `899f55f8e4` | 提供 GLM5Next 的 PyTorch 公式参考，可用于 golden test |

上游 PR 当前仍在快速变动，因此实施时应固定 PR commit 或其最终合入 commit，避免直接依赖浮动分支。

本文三张张量图均以 `XeLaTeX + ctex(fontset=fandol) + TikZ` 绘制，并从同一 `.tex` 源导出 PDF、SVG、透明 PNG 和白底 PNG。

三张图采用多行原生流水线：每个中间 tensor 都标明生产者、shape、后续消费者；不透明的高层模块框已拆为 matmul、broadcast、reduction、mask、softmax、state read/write 等原语。浅色虚线框对应新增/改造算子的实际融合范围。KDA 图中的 OP-KDA-03 与 OP-KDA-04 是运行时互斥的 prefill/decode 路径：03 单独展开 chunk-parallel 累计衰减与下三角变换，04 单独展开逐 token recurrence；Linear 投影及 prefill state gather/scatter 明确位于融合框外。

### 2.3 配套算子需求规格

- [KDA Kernel 需求规格](./glm53_flash_vllm_cl_kernel_specs/01_KDA_kernel需求规格.md)
- [KPool-DSA 与 NoPE 稀疏 MLA Kernel 需求规格](./glm53_flash_vllm_cl_kernel_specs/02_KPool-DSA与NoPE稀疏MLA_kernel需求规格.md)
- [mHC 四路残差 Kernel 需求规格](./glm53_flash_vllm_cl_kernel_specs/03_mHC四路残差_kernel需求规格.md)
- [其他算子改造需求规格](./glm53_flash_vllm_cl_kernel_specs/04_其他算子改造需求规格.md)

## 3. 模型结构账本

### 3.1 主干结构

| 项目 | GLM-5.3-Flash 配置 |
|---|---:|
| 总参数 / 激活参数 | 约 321B / 18B |
| decoder 层数 | 45 |
| hidden size | 4096 |
| Attention 排列 | 34 KDA + 11 KPool-DSA |
| 层模式 | `[KDA, KDA, KDA, DSA] × 11 + KDA` |
| KDA | 64 heads，head dim 128，短卷积 kernel 4 |
| MLA | 64 heads，q LoRA 1536，KV latent 512，q/k NoPE dim 256，RoPE dim 0，V dim 256 |
| DSA indexer | 32 heads，head dim 128，TopK 2048，KPool 4 |
| mHC | 4 residual streams，Sinkhorn 20 次 |
| FFN | 前 3 层 dense；后 42 层 MoE |
| MoE | 288 routed experts，Top-8，1 shared expert，expert width 2048 |
| 激活 | `swiglu_limit=10.0` |
| 上下文 | 1,048,576 tokens |
| MTP | 1 层 |
| 视觉塔 | 24 blocks，hidden 1024，16 heads，patch 14，temporal patch 2，输出 4096 |

### 3.2 各层职责

- 层 0-2：KDA + Dense FFN。
- 层 3：第一层 KPool-DSA + 第一层 MoE。
- 层 4-43：继续每 3 层 KDA 插入 1 层 DSA，均使用 MoE。
- 层 44：KDA + MoE。
- MTP 层 45：DSA + MoE，不使用 mHC；MTP 迭代可复用第一步算出的 TopK。
- 所有 45 个主干层在 attention 和 FFN 两个位置均套 mHC。

## 4. vllm-cl 现状与缺口矩阵

| 能力 | 当前状态 | GLM-5.3-Flash 适配判断 |
|---|---|---|
| GLM5Next 模型注册 | 无 | 必须新增；不能依赖 Transformers fallback，已有案例在 KDA 权重名处失败 |
| 混合 Attention/Mamba 调度 | 已有部分框架 | 可复用 runner 的 Mamba state preprocessing，但需新增 KDA layer/spec 和 KPool tail group |
| KDA | 无 | 核心新增算子与状态 ABI |
| 稀疏 MLA | 已有 BF16 DLC backend | 可复用主体；必须验证/扩展 `qk_rope_head_dim=0`、KV width 512、TopK width 2051/2176 |
| DSA indexer | 已有 token 粒度 BF16 indexer | 不能直接复用；GLM 要池级压缩、池级 TopK、尾池状态，并要求 `ReLU(dot)` 后再按 head 加权 |
| mHC | 无 DLC 专用实现 | 上游 `forward_native` 可用于 bring-up；生产需 `pre/post/fused_post_pre` 融合 |
| MoE | 已有 BF16/AWQ、sigmoid TopK | 基本可复用；必须补 `swiglu_limit=10`，并保留 FP32 router |
| 视觉塔 | Conv2d/3d 和多模态基础存在 | 需要 GLM5Next processor、视觉拓扑、NoPE 文本模型桥接及精度验证 |
| MTP/EAGLE | 有通用补丁 | KDA 状态回滚、KPool TopK 复用和 MTP 层权重映射仍需新增；第一阶段建议关闭 |
| FP8 | `supports_fp8=False` | 默认 FP8 checkpoint 无法作为首个闭环；先用 BF16，或单独建设 block-FP8 loader/算子 |
| Triton / torch.compile | 仓库明确禁止 | 上游 PR 中的 Triton、TileLang 和 `@torch.compile` 路径不能直接搬运 |
| KV block size | 固定 256 | 对 KPool 合法；对上游共享页 KDA/MLA 布局不总是合法，需要 DLC 专用分池分配 |

## 5. KDA：输入、输出与计算逻辑

![KDA 张量流](./glm53_flash_vllm_cl_figures/glm53_kda_tensor_flow.png)

制图文件：[XeLaTeX/TikZ 源码](./glm53_flash_vllm_cl_figures/glm53_kda_tensor_flow.tex) · [PDF](./glm53_flash_vllm_cl_figures/glm53_kda_tensor_flow.pdf) · [SVG](./glm53_flash_vllm_cl_figures/glm53_kda_tensor_flow.svg)。

### 5.1 张量接口

令 `P=TP size`，`H_local=64/P`，`D=128`，当前 step 的有效 token 数为 `N`。

| 张量 | 形状 | dtype | 语义 |
|---|---:|---|---|
| `hidden_states` | `[N, 4096]` | BF16 | mHC collapse 后、RMSNorm 后的子层输入 |
| `q,k,v` | 各 `[N,H_local,128]` | BF16 | 短卷积 + SiLU 后的投影；q/k 在递归前 L2Norm |
| `beta_logits` | `[N,H_local]` | BF16 | delta 写入强度 logit，kernel 内 FP32 sigmoid |
| `z` | `[N,H_local,128]` | BF16/FP32 | 逐头逐通道 forget gate logit |
| `g_out` | `[N,H_local,128]` | BF16 | 输出门，应用于 head-wise RMSNorm 后 |
| `conv_state` | `[num_slots,3H_local·128,3+n_spec]` | BF16 | q/k/v 深度卷积历史；无 MTP 时宽度为 3 |
| `S` | `[num_slots,H_local,128,128]` | FP32 | KDA 递归状态 |
| `core_out` | `[N,H_local,128]` | BF16 | KDA read 结果 |
| `output` | `[N,4096]` | BF16 | gated RMSNorm、flatten、RowParallel `W_o` 后输出 |

KDA 的投影可以先按独立线性层实现，性能版本再像上游 PR 一样融合为一次 GEMM：

- q/k/v：各 `4096 -> 8192`；
- beta：`4096 -> 64`；
- forget/output gate 的 A 路：各 `4096 -> 128`，在 TP rank 间复制；
- B 路：各 `128 -> 8192`。

### 5.2 单 token 递归

对每个 head，先计算：

```text
q_t = L2Norm(SiLU(DWConv4(W_q x_t)))
k_t = L2Norm(SiLU(DWConv4(W_k x_t)))
v_t =          SiLU(DWConv4(W_v x_t))
beta_t = sigmoid(W_beta x_t)
g_t = -5 * sigmoid(exp(A_h) * (z_t + dt_bias_h))
alpha_t = exp(g_t)
```

状态公式采用论文记法 `S ∈ R^(Dk×Dv)`：

```text
S_decay = Diag(alpha_t) S_(t-1)
v_hat   = S_decay^T k_t
delta_v = beta_t (v_t - v_hat)
S_t     = S_decay + k_t delta_v^T
o_t     = S_t^T q_t / sqrt(128)
```

随后：

```text
y_t = W_o(sigmoid(g_out_t) * RMSNorm(o_t))
```

这与矩阵式 `(I-beta*k*k^T) Diag(alpha) S + beta*k*v^T` 等价。

### 5.3 Prefill 与 Decode 必须分开实现

**Prefill** 不能用 Python/token 循环作为正式实现。建议先保留一个短序列 FP32 reference，生产 kernel 采用 chunked KDA：

- chunk 内并行，chunk 间传递 `S`；
- `g` 在 chunk 内做累计和/累计衰减；
- 因 gate 下界为 -5，可在有限数值范围内使用矩阵化 causal tile；
- 输入为 packed token 加 `query_start_loc`，输出最终状态并 scatter 到 request state slot。

**Decode** 需要单次读取并原位写回 `conv_state` 和 `S`。若启用 MTP，kernel 还必须接收 `num_accepted_tokens`，避免 rejected draft token 污染状态；可采用 ReplaySSM 式“缓存投影、验收后片上回放”。

### 5.4 建议的 DLC 算子

| 算子 | 输入 | 输出/副作用 | 优先级 |
|---|---|---|---|
| `dlc_kda_causal_conv_prefill` | qkv projection、DWConv 权重、`query_start_loc`、初始 conv state | qkv conv 输出、最终 conv state | P0 |
| `dlc_kda_causal_conv_update` | 1 或多 draft token、conv state、state indices、accepted counts | conv 输出并原位更新 state | P0 |
| `dlc_kda_chunk_prefill` | q/k/v、raw z、beta logits、A/dt bias、初始 `S`、cu_seqlens | `core_out`、最终 `S` | P0 |
| `dlc_kda_recurrent_decode` | 同上，外加 request state indices | `core_out`、原位 `S` | P0 |
| `dlc_rmsnorm_sigmoid_gate` | `core_out`、`g_out`、norm weight | gated output | P1，可先 native |
| `dlc_kda_fused_decode` | projection 后张量 + conv/S state | 融合 conv、norm、gate、recurrence | P2 性能优化 |

### 5.5 状态内存

无 MTP 时，每个 KDA 层、每个请求的全局状态约为：

```text
recurrent = 64 * 128 * 128 * 4 bytes = 4.000 MiB
conv      = 3 * 64 * 128 * 3 * 2 bytes = 0.141 MiB
total     = 4.141 MiB，TP 后按 head 维约除以 P
```

34 层合计约 `140.8 MiB / request`（全局），TP8 时约 `17.6 MiB / rank / request`。

## 6. KPool-DSA 与 NoPE 稀疏 MLA

![KPool-DSA 张量流](./glm53_flash_vllm_cl_figures/glm53_kpool_dsa_tensor_flow.png)

制图文件：[XeLaTeX/TikZ 源码](./glm53_flash_vllm_cl_figures/glm53_kpool_dsa_tensor_flow.tex) · [PDF](./glm53_flash_vllm_cl_figures/glm53_kpool_dsa_tensor_flow.pdf) · [SVG](./glm53_flash_vllm_cl_figures/glm53_kpool_dsa_tensor_flow.svg)。

### 6.1 KPool 压缩

索引器为每个 token 生成：

| 张量 | 形状 | dtype |
|---|---:|---|
| index query `q_i` | `[Q,32,128]` | BF16 |
| index key `k_i` | `[L,128]` | BF16 |
| head weights `w` | `[Q,32]` | FP32 推荐 |
| gate score `c` | `[L,128]` | BF16/FP32 |
| APE | `[4,128]` | FP32 |

每 4 个 token 形成一个完整池。对池 `p`、池内位置 `r`、通道 `d`：

```text
pi[p,r,d] = softmax_r(c[p,r,d] + APE[r,d])
k_pool[p,d] = sum_r pi[p,r,d] * k[p,r,d]
```

这里 softmax 是逐通道、沿 4 个池内 token 计算，不是对 128 个通道计算。

### 6.2 池级打分、离散索引与展开

正确的索引分数是：

```text
head_score[t,h,p] = ReLU(dot(q[t,h], k_pool[p]) / sqrt(128))
score[t,p] = sum_h (w[t,h] / sqrt(32)) * head_score[t,h,p]
pool_ids[t] = TopKIndices(score[t], 512)
token_ids[t] = flatten([4*p + 0, 4*p + 1, 4*p + 2, 4*p + 3] for p in pool_ids[t])
```

然后附加当前未完成池的最多 3 个 tail token。有效输出宽度最多 `2048+3=2051`，上游实现为稀疏 MLA tile 将缓冲区向上对齐到 2176，并用 `-1` 填充无效槽。

**当前 vllm-cl 的 indexer reference 不能直接作为 GLM golden**：它先对 q 按 head 加权再点积，缺少逐 head 点积后的 ReLU。两者一般不等价。必须确认 `torch.dlc_multi_query_attention` 的真实语义：

- 若硬件 op 已包含 `ReLU(dot)`，修正 CPU reference 和测试；
- 若硬件 op 也按“weighted q 再 dot”，新增 GLM/KPool 专用 MQA logits 算子。

### 6.3 BF16 路径可以省略 Hadamard + FP8

上游 NVIDIA 路径对 q 和 pooled k 同时做归一化 Hadamard-128 旋转并进行 FP8 量化。对于 DLC 的首个 BF16 正确性版本，可让二者都保持未旋转 BF16：

```text
dot(Hq, Hk) = dot(q, k)    because H^T H = I
```

因此 FWHT 不是 BF16 bring-up 的必要新算子；它是后续 FP8 index cache 的性能/容量优化。不能只旋转 q 或只旋转 k。

### 6.4 KPool cache 与 tail cache

- pooled index cache：每 4 token 存 1 个 `[128]` BF16 key；`tokens_per_state=4`。
- tail cache：每请求每 DSA 层保存 `[2,4,128]` BF16，两个平面分别是 raw K 和 gate score。
- pool 完成时先读取此前 tail，再加入当前 token 做压缩；随后把当前 token 写入环形 tail。
- prefill、chunked prefill、decode、MTP verify 和 PD transfer 都必须维持相同的 pool 边界。
- tail cache 不是 prefix-cacheable 的普通 KV block，需要 one-block-per-request 的 circular manager。

固定 `block_size=256` 对 KPool 是合法的：256 可被 4 整除，且 pooled storage block 有 64 个条目。

### 6.5 建议的 DLC 算子

| 算子 | 输入 | 输出/副作用 | 备注 |
|---|---|---|---|
| `dlc_kpool_compress_write` | `[M,4,128]` K、gate、APE、pool slot mapping | `[M,128]` pooled K 写 cache | P0；BF16 版本无需 FWHT |
| `dlc_kpool_tail_seed` | prefill K/gate、tail slots | 写 `[Breq,2,4,128]` | P0 |
| `dlc_kpool_decode_update` | `[B,next_n,128]` K/gate、position、pool/tail slots | tail 原位更新，完成池时写 pooled cache | P0；同请求 token 必须按 position 顺序 |
| `dlc_kpool_mqa_logits` | q `[Q,32,128]`、pooled K、w `[Q,32]`、有效长度 | score `[Q,P]` FP32 | P0；确认 ReLU 语义 |
| `dlc_expand_pool_topk` | pool ids `[Q,512]`、token seq len | token ids `[Q,2051]` int32 | P1，可先 PyTorch |

### 6.6 NoPE 稀疏 MLA

GLM-5.3-Flash 的 `qk_rope_head_dim=0`，主 MLA cache 只保存 512 维 latent：

| 对象 | 形状 |
|---|---:|
| MLA cache | `[num_blocks,256,512]` BF16 |
| absorbed query | `[Q,64/TP,512]` |
| topk token ids | `[Q,2051]`，实现缓冲区可为 `[Q,2176]` |
| sparse output | `[Q,64/TP,512]`，再经 V up-projection 与 `o_proj` |

要逐项验证：

1. `concat_and_cache_mla_dlc` 是否允许空的 `k_pe`，否则增加只写 512 维 latent 的分支；
2. `torch.dlc_sparse_attention_kv_cache` 是否硬编码 576=`512+64`，否则增加 512 维 NoPE kernel variant；
3. 稀疏 kernel 是否接受 2051/2176 宽度和 `-1` padding；
4. decode LSE 返回、DCP index 映射和 block table 转换是否仍正确。

1M 上下文、BF16 时，11 层 MLA cache 约为 `11 GiB / rank / full-context request`；11 层 pooled index cache 约为 `704 MiB / rank`。

## 7. mHC：四路残差混合

![mHC 张量流](./glm53_flash_vllm_cl_figures/glm53_mhc_tensor_flow.png)

制图文件：[XeLaTeX/TikZ 源码](./glm53_flash_vllm_cl_figures/glm53_mhc_tensor_flow.tex) · [PDF](./glm53_flash_vllm_cl_figures/glm53_mhc_tensor_flow.pdf) · [SVG](./glm53_flash_vllm_cl_figures/glm53_mhc_tensor_flow.svg)。

### 7.1 输入输出

令 `N` 为 token 数，`Hc=4`，`D=4096`，`mix=(2+Hc)Hc=24`。

| 张量 | 形状 | dtype |
|---|---:|---|
| residual streams `X` | `[N,4,4096]` | BF16 |
| `fn` | `[24,16384]` | FP32 |
| `base` | `[24]` | FP32 |
| `scale` | `[3]` | FP32 |
| `pre` | `[N,4,1]` | FP32 |
| `post` | `[N,4,1]` | FP32 |
| `comb` | `[N,4,4]` | FP32 |
| collapsed sublayer input | `[N,4096]` | BF16 |
| updated streams | `[N,4,4096]` | BF16 |

### 7.2 计算

```text
flat = X.reshape(N, 16384).float()
mix_logits = (flat @ fn^T) * rsqrt(mean(flat^2) + rms_eps)

pre  = sigmoid(mix_logits[:,0:4]   * scale[0] + base[0:4]) + eps
post = 2 * sigmoid(mix_logits[:,4:8] * scale[1] + base[4:8])
C     = mix_logits[:,8:24].reshape(N,4,4) * scale[2] + base[8:24]
comb  = Sinkhorn20(softmax(C, dim=-1) + eps)

x_in = sum_i pre_i * X_i
X_out_j = post_j * f(x_in) + sum_i comb_ij * X_i
```

第一层先把 `[N,4096]` 复制为四路；最后一层对 stream axis 取均值。GLM-5.3-Flash 不使用 DeepSeek-V4 的 gated HC head。

### 7.3 是否必须新增 mHC kernel

- **正确性 bring-up**：不是绝对必须写单体 kernel。可以复用 `mhc_pre_torch` / `mhc_post_torch` 的 PyTorch 计算，但 OOT wrapper 必须在 pre-collapse 后显式补上模型传入的子层 RMSNorm；上游 `mhc_pre_torch` 本身不处理 `norm_weight/norm_eps`。
- **生产性能**：必须优化。45 层 × 2 个 mHC site，每次都有 residual 读写和 20 次 Sinkhorn，小 op launch 和 HBM 流量会非常明显。

建议提供三个 OOT op：

| OOT 类/算子 | 功能 |
|---|---|
| `MHCPreOp` / `dlc_mhc_pre` | RMS、24 维映射、pre/post/comb、Sinkhorn、collapse |
| `MHCPostOp` / `dlc_mhc_post` | `comb^T * residual + post * sublayer_out` |
| `MHCFusedPostPreOp` / `dlc_mhc_fused_post_pre` | 上一 site 的 post 与下一 site 的 pre/RMSNorm 融合，减少四路 residual 往返 |

## 8. MoE、激活和路由的隐藏正确性问题

### 8.1 SwiGLU clamp

GLM-5.3-Flash 所有 dense FFN、shared expert、routed expert 和视觉 MLP 都要求：

```text
gate = clamp(gate, max=10)
up   = clamp(up, min=-10, max=10)
out  = SiLU(gate) * up
```

当前 DLC fused MoE 只声明 `MoEActivation.SILU`，调用接口没有显式传递 `swiglu_limit`。如果直接复用会产生模型输出偏差。需要：

- 扩展 DLC monolithic MoE kernel 支持 clamp；或
- 首阶段对 GLM5Next 关闭 monolithic fused path，使用正确的 reference 路径；
- 为 `SiluAndMulWithClamp` 增加 DLC OOT fused op，dense/shared/vision 路径都可复用。

### 8.2 Router 完整精度目标与首阶段 BF16 决策

checkpoint 设置 `moe_router_dtype=float32`，Transformers reference 也在 FP32 计算 router logits。
当前 GLM5 构造 `GateLinear` 时传入 `out_dtype=torch.float32`，但 AWQ
`DLCAWQMarlinMoEMethod.apply_monolithic` 在 fused MoE 调用前仍把 logits 转为 BF16，
底层 `dlc_vllm_fused_moe_w4a16_awq_col_major_impl` 也显式要求 BF16 gating input。
因此仅保留 `GateLinear.out_dtype` 不能闭合 FP32 routing 链路。插件的
`DLCGateLinear.set_out_dtype()` 另有强制 BF16 逻辑，但 GLM5 构造时已设置 dtype，
不能把该 setter 必然执行当作判断依据。

**2026-09-15 阶段决策：第一阶段暂时接受 BF16 routing logits，用于真实服务启动和少量请求测试；
FP32 router 归入后续算子接口工作，不作为首次启动前的代码阻断，本轮不改接口。**
BF16 舍入可能改变临界 expert 的 Top-8 和路由权重，正确性对比需要记录这一已接受的精度差异。
该决策仅放宽第一阶段要求，完整精度目标仍为：

- 当模型/层请求 FP32 router 时尊重 `out_dtype=torch.float32`；
- 扩展 fused MoE 的 FP32 logits 输入及 routing 实现，或提供 native FP32 routing 与 expert 计算分离的调用接口；
- top-8 选择、`e_score_correction_bias`、归一化和 `routed_scaling_factor=2.5` 以 FP32 完成；
- `n_group=1`，因此无需分组裁剪，但仍需 sigmoid score、correction bias 和 Top-8。

## 9. KV cache 与固定 block-size 256

### 9.1 KPool 条件满足

上游 KPool 要求 block size 为 `index_kpool * 32 = 128` 的倍数；DLC 的 256 满足要求，池化后每个 block 对应 64 个 pooled entries。

### 9.2 上游共享页策略在 DLC 上会卡住

上游 PR 尝试把 KDA state page padding 到 MLA page。DLC BF16 MLA page 大小为：

```text
256 tokens * 512 dims * 2 bytes = 256 KiB
```

KDA state 每层每请求、每 rank 的无 MTP 大小约为：

| TP | KDA state / layer / rank | 是否放得进 256 KiB MLA page |
|---:|---:|---|
| 4 | 1.035 MiB | 否 |
| 8 | 0.518 MiB | 否 |
| 16 | 0.259 MiB | 否，略超 |
| 32 | 0.129 MiB | 是 |

因此不能照搬 `_get_kv_cache_groups_glm5_next` 的共享页假设，否则固定 block 256 会把最低 TP 推到 32。

### 9.3 推荐的 DLC 分池方案

在 `patch_kv_cache_utils.py` 增加 GLM5Next 特殊路径：

1. scheduler 仍使用统一逻辑 block id 和 256-token 边界；
2. MLA、pooled index、KPool tail、KDA state 使用独立物理 tensor/pool；
3. 所有 pool 采用相同 `num_blocks`，但每层 page bytes 按各自 spec 计算，不强行 padding 到 MLA page；
4. KPool tail 使用 one-block-per-request manager，不参与 prefix hash；
5. 容量估计按各组真实 page bytes 求和；
6. prefix hit 必须同时满足 MLA block、KDA state checkpoint 和全部 KPool tail/index group 的一致边界。

`vllm-cl` 已有“不同 group 单独 tensor、统一 num_blocks”的代码基础，可复用该思路，但要把 `KpoolTailSpec` 和 `tokens_per_state=4` 纳入 shape、manager 和容量计算。

**2026-09-17 当前实现状态：**V1/V2 DLC runner 已支持为 KDA `MambaSpec` 独立分配
`int8 [num_blocks,1,1,page_size_bytes]` 原始字节页，按该 spec 的完整页宽（包括
`page_size_padded`）计入容量；绑定时由上游 `MambaBase.bind_kv_cache()` 拆出 BF16
卷积状态与 FP32 递归状态。CPU 小张量验证了 V1/V2 混合 attention/Mamba 分组、
状态 view 与原始字节页共享存储，以及跨页 padding stride。此前 offset-free allocator
拒绝 `MambaSpec` 的启动阻断已修复；上述分池方案的完整服务路径及 DLC 实机行为仍待验证。

## 10. 多模态适配

需要引入上游 PR 的 GLM5Next processor 和视觉模型拓扑，不能只复用 GLM4.1V 类名：

- Conv3D patchify：kernel/stride `(2,14,14)`；
- 24 个 vision blocks，hidden 1024，16 heads；
- Q/K RMSNorm + 2D rotary position；
- post RMSNorm 后做 2×2 Conv2D downsample；
- merger：projection、LayerNorm、GELU、clamped SwiGLU，输出 4096；
- 图像和视频 processor 使用 token budget、smart resize、frame sampling、prompt placeholder 展开。

已有 `patch_conv.py` 会让 Conv2d/3d 走 DLC native conv，可复用；仍需验证可变长度 vision attention、rotary、merge shape 及图片/视频 placeholder 数量。

一个必须保留的上游修正：视觉塔实际推理应使用 `rms_norm_eps=1e-6`，不能直接采用 checkpoint `vision_config` 中的 `1e-5`，否则图像描述会明显漂移。

## 11. MTP 适配

MTP 不是首个正确性闭环的必选项。基础模型通过后再实现：

1. 注册 `Glm5NextMTPModel`；
2. MTP 层 45 走 DSA+MoE、不走 mHC；
3. `eh_norm`：对 position 0 的 embedding 置零，分别 RMSNorm embedding 与 previous hidden，concat 为 `[N,8192]`，再投影到 4096；
4. `index_share_for_mtp_iteration=true`：第一 draft step 计算 TopK，后续 step 复用并按 slot compaction；
5. KDA conv/recurrent state 在 verify/reject 后按 accepted token 精确前进；
6. MTP 权重前缀从多模态 checkpoint 的 `model.language_model.layers.45.*` 映射到 text-only draft model。

上游 `fused_eh_norm.py` 是 Triton，DLC 初期可用 `where + 2×RMSNorm + cat`，后续再做融合。

## 12. 权重与量化

### 12.1 第一阶段建议 BF16

默认 `zai-org/GLM-5.3-Flash` 是 block-FP8，expert/dense MLP 权重包含 `weight_scale_inv`；KDA、mHC、视觉塔、norm、router 和若干 MLA 投影保留 BF16/FP32。当前 DLC 平台返回 `supports_fp8=False`，不能直接宣称支持默认 checkpoint。

首个实机闭环建议使用 `zai-org/GLM-5.3-Flash-BF16`：纯权重约 642 GB，按每卡约 60 GB 可用内存估算，至少需要 12 卡才能只装下权重；考虑 KV、状态、激活和框架开销，实际应优先按 16 卡或更高 TP/EP 规划。

### 12.2 后续 FP8 支持

若目标是默认 FP8 checkpoint，需要补齐：

- block-FP8 linear/MoE 权重加载与 scale layout；
- FP8 GEMM/MoE；
- fused parameter name 与 `modules_to_not_convert` 的映射；
- 对必须 BF16 的 attention/indexer 投影做 load-time dequant；
- KDA、mHC、vision 不得误套全局 FP8 quant config。

上游 loader 还处理了以下融合映射，应完整移植或在插件模型中复现：

```text
gate_proj + up_proj               -> gate_up_proj
q_a_proj + kv_a_proj_with_mqa     -> fused_qkv_a_proj
indexer.wk + indexer.weights_proj -> wk_weights_proj
KDA q/k/v/b/f_a/g_a               -> in_proj_qkvbfg_a
```

## 13. 推荐代码组织

由于上游 PR 的 `glm5next/__init__.py` 会直接导入 NVIDIA 实现，而 NVIDIA 文件又在 import 时依赖 Triton/DeepGEMM，当前 vllm-cl 不能只做运行时 monkeypatch。推荐使用插件自有懒加载模型：

```text
vllm_cl/models/glm5next/
├── __init__.py
├── model.py          # topology、weight loading、registry interfaces
├── kda.py            # DLC KDA layer
├── attention.py      # KPool indexer + NoPE sparse MLA
├── multimodal.py     # vision tower / processor bridge
└── mtp.py            # optional

vllm_cl/ops/
├── kda.py
├── kpool.py
├── mhc.py
└── sparse_attn_indexer_kpool.py
```

插件初始化时用 `ModelRegistry.register_model()` 覆盖：

- `Glm5NextForCausalLM`
- `Glm5NextForConditionalGeneration`
- `Glm5NextMTPModel`

并修改：

- `vllm_cl/patch/models/__init__.py`
- `vllm_cl/ops/__init__.py`
- `vllm_cl/_custom_ops.py`
- `vllm_cl/patch/patch_kv_cache_interface.py`
- `vllm_cl/patch/patch_kv_cache_utils.py`
- `vllm_cl/patch/patch_mla_indexer.py`
- `vllm_cl/attention/sparse_mla.py`
- `vllm_cl/ops/activation.py`
- `vllm_cl/ops/fused_moe.py`
- `vllm_cl/ops/gate_linear.py`
- `vllm_cl/patch_import_manifest.yaml`
- `vllm_cl/vllm_alignment.yaml`

不要把上游 `nvidia/ops/*.py` 或 FLA Triton 路径原样复制到插件。

## 14. 分阶段实施路线

### Phase 0：源码对齐

- 把 vllm-cl 对齐到包含 GLM5Next cache/spec/processor 基础设施的固定 vLLM commit；
- 记录 PR #53906 或最终合入 commit；
- 更新 patch manifest，确认旧 GLM/DeepSeek patch 未因符号变更失效。

### Phase 1：BF16 文本正确性，无 MTP

- 注册 text model；
- KDA short-conv + recurrent reference/kernel；
- mHC native reference；
- BF16 KPool cache/tail/indexer；
- NoPE sparse MLA；
- MoE clamp；router 首阶段暂用 BF16，FP32 算子接口工作后续完成（见 8.2）；
- DLC 独立 KV/state 分池；
- 关闭 PP、MTP、FP8、prefix caching，先完成短文本推理。

### Phase 2：长前缀与性能

- chunked KDA prefill kernel；
- fused KDA decode；
- fused mHC；
- KPool compress/tail/TopK 融合；
- chunked prefill、prefix caching、DCP/CP；
- 逐步把 `max_model_len` 从 4K、32K、128K 拉高。

### Phase 3：多模态

- 图像 processor + vision tower；
- 视频采样与 placeholder；
- text+image、短视频请求；
- 检查 vision RMSNorm epsilon 与视觉特征 finite。

### Phase 4：量化与 MTP

- FP8 或 AWQ 路径；
- MTP layer、TopK reuse、KDA replay/rollback；
- PD transfer 和混合 cache prefix reuse；
- 性能和长稳测试。

## 15. 测试与验收矩阵

### 15.1 Kernel/unit

| 测试 | 必须覆盖 |
|---|---|
| KDA recurrence | 单步 vs Transformers FP32 reference；整段 prefill vs 分段 prefill+continue；prefill 后逐 token decode |
| KDA cache | 多请求 state indices；新请求清零；request 重排；MTP accepted/rejected（启用 MTP 后） |
| causal conv | kernel=4；跨 chunk 历史；position 0；多 draft token |
| KPool | 1-3 token tail、恰好 4、跨 256 block、chunk 边界 255/256/257、多请求、左 padding |
| index score | 明确验证 `sum_h w_h*ReLU(dot_h)`，TopKValues 与 TopKIndices 分离 |
| NoPE sparse MLA | 512 维 KV、空 PE、2051/2176 indices、全 `-1` 行、decode LSE |
| mHC | pre/post/fused 与 `mhc_*_torch` 对齐；comb 行列和接近 1；最后 stream mean |
| MoE | FP32 router Top-8 ID 完全一致；correction bias；shared expert；SwiGLU clamp |
| weight loader | BF16；融合权重；默认 FP8 仅在对应能力完成后开放 |

### 15.2 Dummy 模型

`num_hidden_layers=4` 正好覆盖 `[KDA,KDA,KDA,DSA]`，并同时覆盖前 3 层 dense 与第 4 层 MoE，是合适的最小 dummy 拓扑。仍需单独构造 vision dummy。

### 15.3 真权重服务验收

所有启动命令都必须显式包含 `--block-size 256`，并确认日志中没有 Triton、TileLang、`torch.compile` 或 `torch._dynamo` 路径。

依次要求：

1. `GET /v1/models` 成功；
2. `/v1/completions` 返回非空文本；
3. `/v1/chat/completions` 返回非空文本；
4. 长前缀触发 chunked prefill，服务不退出；
5. 同一 prompt 与 Transformers/vLLM reference 做 greedy token/logit 对齐；
6. text+image 成功；
7. video 成功；
8. MTP 在用户选择启用后单独验收；
9. 长稳期间检查 KDA state、KPool tail、prefix hit 后的重复/漂移/非有限值。

## 16. 已知风险与必须实机确认项

1. `torch.dlc_multi_query_attention` 是否实现逐 head ReLU 后加权；源码不在本仓库，必须 probe。
2. `torch.dlc_sparse_attention_kv_cache` 是否支持 NoPE 512 维和 2176 index width；必须 probe。
3. DLC PyTorch 的 FP32 matmul/softmax/einsum 能否承载 mHC reference；正确性可能可以，性能未确认。
4. 固定 256 block 下的 KDA/Mamba state 必须采用分池布局，否则 TP<32 会被共享页尺寸卡死。
5. vllm-cl 当前对齐版本落后于本地 vLLM main；先做 main-to-main 对齐，否则后续补丁结论不稳定。
6. 上游 GLM PR 仍未合入且近期 bug 修复频繁；实施必须固定 commit，并把已知 upstream 修复逐项审计后再移植。
7. 当前文档未在 DLC 真机加载 321B 权重，因此性能、最大 TP/EP 拓扑和数值容差需要硬件验证。

## 17. 最小可交付定义

只有同时满足以下条件，才能称为“GLM-5.3-Flash 已适配 vllm-cl”：

- 原生 `Glm5NextForConditionalGeneration` 注册成功；
- KDA prefill/decode 与状态缓存正确；
- KPool-DSA 在 chunk/block/tail 边界正确；
- NoPE sparse MLA 正确；
- mHC、clamped SwiGLU、FP32 router 正确；
- BF16 真权重 completion/chat/长前缀通过；
- 图片请求通过；
- 全流程无 Triton/`torch.compile`；
- regression config 和 kernel golden tests 已进入仓库；
- FP8、MTP、视频、prefix cache、DCP/PP 若未完成，必须明确标为未支持，不能隐含宣称。

## 参考链接

- [GLM-5.3-Flash Hugging Face](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [vLLM PR #53906: add GLM-5.3-Flash support](https://github.com/vllm-project/vllm/pull/53906)
- [Transformers GLM5Next reference](https://github.com/huggingface/transformers/blob/899f55f8e41654a106a414e93a6f5fd8f3b8bf37/src/transformers/models/glm5_next/modeling_glm5_next.py)
- [NVIDIA NeMo GLM-5.3-Flash](https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/thudm/glm-5-3-flash)
