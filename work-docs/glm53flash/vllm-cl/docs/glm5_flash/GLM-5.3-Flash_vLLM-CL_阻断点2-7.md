# GLM-5.3-Flash 在 vLLM + vLLM-CL 中的阻断点（2–7）

本文记录当前 vLLM 与 vLLM-CL 工作树中，GLM-5.3-Flash 适配的第 2–7 项阻断点。
第 2、3、4、5、6、7 项已有代码或静态验证结果；本阶段只做源码和 CPU 小张量验证，未启动模型服务，也未加载真实权重。

目标 checkpoint 的关键配置是：34 个 KDA 线性注意力层、11 个 KPool-DSA/NoPE 稀疏 MLA 层、4 路 mHC、42 个 AWQ MoE 层；`swiglu_limit=10.0`、`kv_lora_rank=512`、`qk_rope_head_dim=0`、`index_kpool=4`、`index_topk=2048`。

## 2. GLM5 KDA native 初稿（调用链验收待完成）

### 代码路径

修复前，`vllm/models/glm5next/nvidia/kda.py` 在模块导入时直接导入：

```text
vllm.models.glm5next.nvidia.ops.third_party.kda
  -> chunk_kda_with_fused_gate
  -> fused_recurrent_kda
```

这两个函数位于 [`vllm/vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py`](/root/workspace/vllm/vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py)，实现中包含 `@triton.jit` kernel。GLM5 KDA 前向在 `kda.py` 的 prefill 和 decode 分支直接调用这两个符号。

vLLM-CL 的 [`patch_fused_recurrent.py`](/root/workspace/vllm-cl/vllm_cl/patch/patch_fused_recurrent.py) 只替换通用 FLA/Qwen GDN 模块；其 fallback 在第 119 行明确断言 `not is_kda`。[`vllm_cl/ops/gdn_linear_attn.py`](/root/workspace/vllm-cl/vllm_cl/ops/gdn_linear_attn.py) 注册的 OOT 层也是 `QwenGatedDeltaNetAttention`，不会替换 `Glm5NextLinearAttention`。DLC backend 映射也没有注册 `GDN_ATTN` 专用的 GLM5 替换层。

虽然 [`vllm_cl/platform.py`](/root/workspace/vllm-cl/vllm_cl/platform.py:15) 会在 decorator 执行前把 `torch.compile` 替换为 identity，但这只能去掉编译装饰器，不能把 GLM5 KDA 中的 Triton kernel 转换为 DLC 算子。当前插件已替换 KDA 的递推、短卷积、状态 gather/scatter、门控 RMSNorm 和 sigmoid helper；NVIDIA 模块仍会在导入阶段解析其 Triton 符号，但 native 前向不会调用这些 kernel。若要连导入依赖也移除，需要后续拆分模型模块。

### 当前处理结果

新增 [`patch_kda.py`](/root/workspace/vllm-cl/vllm_cl/patch/patch_kda.py)，在插件加载阶段
重绑定 GLM5 KDA 模块实际使用的 `fused_recurrent_kda` 和 `chunk_kda_with_fused_gate`，并替换其短卷积、状态 gather/scatter、输出 gated RMSNorm 与 sigmoid helper。递推在 FP32 中执行 safe gate、per-head beta、q/k L2 normalization、delta rule 和最终状态写回；状态按 metadata 选定 page 读写，不再整块复制 cache。native 卷积按请求逐序列更新 `[dim,width-1]` 历史，RMSNorm 在 FP32 中完成归一化和 sigmoid 门控。普通 prefill/decode 使用与 NVIDIA 调用侧相同的布局和 state index；speculative state page、residual/prenorm RMSNorm 目前显式报 `NotImplementedError`，本阶段关闭 MTP/speculative。

新增 [`test_kda_native.py`](/root/workspace/vllm-cl/tests/test_kda_native.py)，使用 CPU 小张量检查单/多 token recurrence、非零 page、L2 normalization、卷积 page 更新、gated RMSNorm 及 speculative 状态保护。

### 修复前影响

- DLC 不支持 Triton 时，GLM5 KDA 不具备可执行的 DLC prefill/decode 实现；当前已有插件 native 初稿，但完整 caller 尚未完成设备级和 reference 等价性验收。
- 通用 `chunk_gated_delta_rule` 或 `fused_recurrent_gated_delta_rule` 补丁不能自动解决 GLM5 KDA，因为 GLM5 使用的是独立的 KDA API、gate 公式和状态布局。
- KDA 状态包含每层 recurrent state 与短卷积 state；不能简单退化到只支持普通 GDN 的路径。

### 剩余工作与验收标准

按 bring-up 到性能优化分两步：

1. 在 DLC 设备上验证插件重绑定后的 GLM5 KDA prefill/decode，确认 native 实现不调用 Triton，并确认 conv/recurrent cache 的真实 layout。
2. 用 HF/NVIDIA reference 对照 q/k L2 normalize、safe gate、beta、delta rule、输出和最终状态；普通 2D 多请求 decode 的 native 卷积更新已修复，并由 CPU 回归覆盖 batch、state page 重排和输出 buffer。剩余边界包括 `initial_state=None` 多页、非法索引写前校验、speculative conv 写前保护、`safe_gate=False` 语义、`output_final_state=False` 以及 profile/padding 输出初始化。卷积权重已通过显式跳过 col-major 后处理标记保留原始卷积布局。
3. 再把 native recurrence 中的循环替换为 DLC 融合原语或新建 KDA kernel；spec decode、状态回滚和 MTP 放在基础路径正确后处理。

需要特别核对 q/k L2 normalize、`beta=sigmoid(beta_logits)`、`g=-5*sigmoid(exp(A)*z)`、状态读写布局，以及 `conv_state` 的 TP 分片。

### 验收标准

- native 数值计算入口已替换，但尚未证明完整 GLM5 caller 在所有 decode/prefill 分支均不调用 Triton；模型模块导入仍会解析 NVIDIA Triton 符号。
- prefill 与 decode 都能在同一 KDA 层上完成状态连续性测试。
- 单 token decode 与短序列 reference 的 `core_out`、最终状态误差在预设 BF16 容差内。

## 3. MLA 调用侧、插件执行路径与稀疏 backend 不匹配（代码已处理，设备验证待做）

修订日期：2026-09-11。目标执行路径采用 **插件中的 GLM5 MLA 调用类 → 当前自研优化版通用 MLA wrapper → 插件中的 512 维 native sparse attention**。GLM5 MLA 的构造与 forward 使用插件类，避免执行 `vllm/models/glm5next/nvidia/attention.py` 中的对应实现；通用基础层可以继续复用主仓库已有的 DLC 适配。

本轮已落地的代码位于 [`vllm_cl/patch/models/patch_glm5next.py`](/root/workspace/vllm-cl/vllm_cl/patch/models/patch_glm5next.py)、[`vllm_cl/attention/sparse_mla.py`](/root/workspace/vllm-cl/vllm_cl/attention/sparse_mla.py)，并由 [`tests/test_glm5next_dlc_contract.py`](/root/workspace/vllm-cl/tests/test_glm5next_dlc_contract.py) 做 CPU 小张量契约验证。512 native 文件已移除未使用的 Triton 导入；KPool factory 现在接入受限的 native caller，真实设备路径仍待验证。

### 3.1 `fuse_qkv_rmsnorm` 构造参数缺失

GLM5 当前在 [`attention.py`](/root/workspace/vllm/vllm/models/glm5next/nvidia/attention.py:569) 构造 `MultiHeadLatentAttentionWrapper` 时传入：

```python
fuse_qkv_rmsnorm=True
```

当前 [`vllm/vllm/model_executor/layers/mla.py`](/root/workspace/vllm/vllm/model_executor/layers/mla.py:53) 的 `MultiHeadLatentAttentionWrapper.__init__` 没有这个参数，也没有 `**kwargs`。因此模型初始化静态上会报 `TypeError: unexpected keyword argument 'fuse_qkv_rmsnorm'`。

历史证据如下（以当前 HEAD 的祖先提交为准）：

| Commit | `mla.py` 的变化 |
|---|---|
| `98ed0856f3` | GLM5 支持加入 `fuse_qkv_rmsnorm` 和融合 q/kv RMSNorm 路径 |
| `0752a8e651` | 自研 reshape/copy 优化提交删除上述参数和融合调用，改用独立 q-A/kv-A RMSNorm；同时删除 g_proj、DCP qrep 等分支，并改变 rotary 输入组织 |
| `02ddd8915b` | 恢复拆分的 query RoPE slice 与 positional key 输入，适配 DLC rotary 接口 |
| `9bf0f48a88` | 恢复 `non_causal_multi_token_decode` 等兼容参数，未恢复 `fuse_qkv_rmsnorm` |

所以这里是 GLM5 调用侧与自研优化版 wrapper 的 API 不一致，不能归因于尚未合入 GLM5 对应的 `mla.py`。其中 `g_proj` 等被删除功能属于其他模型/并行路径，不能因存在历史差异就全部恢复到本模型适配中。

`glm-5.3-flash/tmp/vllm-pr53906/vllm/model_executor/layers/mla.py` 中的旧实现可以作为语义参考，但不能直接覆盖当前 `mla.py`：它会重新引入已经被自研优化删除的 reshape/copy 路径。这里记录的是修复前状态；当前外层 wrapper 已由 vLLM-CL 提供。

GLM5 的 q-B projection 必须使用 q-A RMSNorm 之后的结果，KV cache 必须使用 kv-A RMSNorm 之后的 latent。当前独立 norm 路径保留这两个数学步骤，融合开关是实现选择；插件调用侧可以去掉该参数，而不是恢复上游融合 API。

### 3.2 目标执行路径应由 vLLM-CL 接管

修复前，`Glm5NextMLAAttention` 是普通 `nn.Module`，不是可直接通过 `PluggableLayer.register_oot` 替换的层；它内部再实例化可替换的 `MultiHeadLatentAttentionWrapper`，所以运行会进入主仓库 NVIDIA 编排代码。当前已由插件在模型实例化前重绑定 `Glm5NextMLAAttention`，并同步覆盖 MTP 模块引用。

推荐实施方式：

1. 在插件中新增 `DLCGlm5NextMLAAttention`，完整实现该类的构造和 forward。维持现有 projection、norm、indexer 的参数名及 TP/weight-loader 协议，不调用 NVIDIA 类的 `super().__init__` 或 `forward`。
2. 通过插件现有 `patch/models` 机制，在模型实例化前，将 `Glm5NextDecoderLayer` 实际引用的 `Glm5NextMLAAttention` 符号绑定到插件类。只替换原始 `attention` 模块的符号不够：`model.py` 已通过 `from ... import ...` 保存了引用。安装过程应幂等，并在模型检查子进程与 worker 中一致执行。
3. 插件类调用当前 `MultiHeadLatentAttentionWrapper` 时去掉 `fuse_qkv_rmsnorm=True`，复用两个独立 RMSNorm 及自研布局路径。不全局替换所有模型的 MLA wrapper；如 NoPE 需要额外消除空 RoPE 拼接，在插件中增加仅用于 GLM5 的 wrapper。
4. KPool indexer 通过插件的 `build_glm5_indexer` 工厂隔离。当前工厂只接入 DCP=PCP=1、`index_kpool=4` 的 native caller，其他并行布局显式报错，绝不回落到 NVIDIA `Indexer`；真实设备构造和 decode 连续性仍需后续验证。

这保证 GLM5 MLA 的运行时类来自插件。该方式允许主仓库模型模块因注册而被导入；“禁止 NVIDIA MLA 构造/forward”与“禁止模块导入”是不同范围，后者还需要独立的插件模型注册入口。

### 3.3 DLC sparse MLA 的 head size capability 不覆盖 GLM5

修复前 [`vllm_cl/attention/sparse_mla.py`](/root/workspace/vllm-cl/vllm_cl/attention/sparse_mla.py:134) 的 `PyTorchMLASparseBackend.get_supported_head_sizes()` 只返回 `[576]`，不覆盖 GLM5 的 512 维 latent cache。

GLM5 NoPE MLA 的缓存 head size 是：

```text
kv_lora_rank + qk_rope_head_dim = 512 + 0 = 512
```

因此即使解决调用侧，selector/backend capability 仍需确认并补上 512；当前 DLC selector 会直接按 `(use_mla, use_sparse)` 选择类，某些校验可能被绕过，但 capability 声明仍不完整。

现有 `PyTorchMLASparseImpl` 名称虽然包含 PyTorch，但 `_forward_bf16_kv` 实际调用 `dlc_ops.sparse_attention`，不能只将 capability 改为 `[512, 576]` 就认为已有 512 支持。

当前插件已新增真正的 native PyTorch 组合路径，tensor 保持在 DLC 设备上；CPU reference 仅用于数值对照。512 分支使用该实现，现有 576 路径继续使用原算子。512 cache 写入和逻辑位置到物理 slot 的转换也使用 native 索引，避免把 512 cache 传入旧 fused kernel。后续专用 512 kernel 按同一接口替换。

### 3.4 512 维 native 实现的计算协议

必须区分 query projection 的 256 维与 latent cache 的 512 维。令 `H=64/TP`：

| 张量 | 形状 |
|---|---|
| q-B projection 输出 `q_nope` | `[N,H,256]` |
| 归一化 KV latent `C` | `[N,512]` |
| 吸收 K 投影后的 `q_latent` | `[N,H,512]` |
| cache | 逻辑 `[num_blocks,256,512]`，实际 stride 由分配器决定 |
| sparse attention latent 输出 | `[N,H,512]` |
| V 投影输出 | `[N,H,256]` |

核心公式：

```text
q_latent = q_nope @ W_UK_T
selected_C = gather(cache, physical_slots(topk, request_id, block_table))
scores = q_latent @ selected_C.T * (256 ** -0.5)
prob = softmax(mask(scores))
latent_out = prob @ selected_C
head_out = latent_out @ W_UV
output = o_proj(flatten_heads(head_out))
```

注意 scale 保持训练时的 `1/sqrt(256)`，不能因 cache/backend head size 是 512 而改为 `1/sqrt(512)`。通用 `MLAAttention` 已提供 q 投影吸收和 V 投影步骤，native sparse backend 只负责中间 cache gather、logits、softmax、latent 聚合，不重复执行这些投影。

实现要点：

- 用 native indexing/index-copy 完成 NoPE cache 写入和读取，尊重 tensor stride、storage offset 与 slot mapping；避免未经检查地 `view` 非连续 cache。
- NoPE 分支直接使用 `q_latent` 和 512 维 cache，跳过零宽 RoPE 的 concat、einsum。
- 在 gather 前校验/屏蔽 `-1`、padding、越界和未来 token；处理全无效行，输出 0，需要 LSE 时返回 `-inf`。
- logits/softmax 先采用 FP32；输出转换为调用方 dtype。提供与现有接口一致的 output/LSE。
- 按 query 分块，避免一次构造所有 token 的 `[N,topk,512]` 和 `[N,H,topk]` 临时张量。
- `W_UK_T/W_UV` 的生成继续遵守插件的 col-major 权重还原逻辑，不能把物理打包权重直接当普通矩阵。

### 影响

- 当前路径尚未到真正权重加载阶段就可能在 MLA 模型构造时失败。
- 直接恢复旧 upstream wrapper 会破坏当前 reshape 优化，或产生形状正确但数值错误的 q/k 投影。
- backend 的 576/512 不匹配会使 NoPE 稀疏 MLA 无法在所有 selector/KV-cache 校验路径中合法选择 DLC backend。

### 处理建议与验收标准

- 检查真实实例的类型/方法来源位于 `vllm_cl`；GLM5 MLA 不调用 NVIDIA 原类的构造或 forward。
- 用固定 TopK 隔离阻断点 7，覆盖 NoPE/q-LoRA、cache 读写、prefill→decode 状态连续性、多请求 block table、无效 TopK 与 LSE。
- 对比 latent 形式与显式展开 K/V 的 FP32 reference，检查投影顺序、scale 与输出形状；再做 DLC BF16 容差检查。
- 确认 512 分支未调用现有 fused sparse attention 或未验证的 512 cache kernel。
- 本项完成只表示 MLA 模块正确，KDA/mHC/AWQ/KPool 的其他阻断点仍需分别完成，尚不代表模型服务可用。

## 4. mHC native 路径忽略 RMSNorm 参数（代码已处理，设备验证待做）

### 代码路径

GLM5 decoder layer 在 attention 和 FFN 子层均使用 `MHCPreOp`/`MHCFusedPostPreOp`，并传入 `norm_weight` 与 `norm_eps`。

当前 [`vllm/vllm/model_executor/layers/mhc.py`](/root/workspace/vllm/vllm/model_executor/layers/mhc.py:210) 的 `MHCPreOp.forward_native` 调用 `mhc_kernels.mhc_pre_torch(...)` 时没有使用这两个参数；`MHCPostOp` 本身不接收 norm 参数，实际缺陷是 Pre/FusedPostPre 返回的 `layer_input` 在 native 路径中未应用 RMSNorm。

PyTorch 仓库已有 DLC 原语：

- [`pytorch/aten/src/ATen/native/dlc/mhc_pre.cxx`](/root/workspace/pytorch/aten/src/ATen/native/dlc/mhc_pre.cxx:73)
- [`pytorch/aten/src/ATen/native/dlc/mhc_post.cxx`](/root/workspace/pytorch/aten/src/ATen/native/dlc/mhc_post.cxx:52)
- `aten::dlc_mhc_pre`、`aten::dlc_mhc_post` 已在 `native_functions.yaml` 注册。

但 vLLM-CL 没有发现针对 `MHCPreOp`/`MHCPostOp` 的 OOT 注册。DLC 若沿用默认 native dispatch，就会走忽略 norm 参数的实现。

### 影响

- 每个 GLM5 主干层的 attention/FFN mHC 输入可能漏掉要求的 RMSNorm。
- 这是确定性的模型正确性风险，不只是性能问题。
- 直接调用已有 `torch.ops.dlc_mhc_pre/post` 也要确认其输出 shape、`hc_mult=4`、Sinkhorn 参数和 norm 的边界顺序。

### 当前处理结果

新增 [`patch_mhc.py`](/root/workspace/vllm-cl/vllm_cl/patch/patch_mhc.py)，为 DLC OOT
dispatch 包装 `MHCPreOp` 与 `MHCFusedPostPreOp` 的 native 调用：保留原有 mHC
FP32 混合计算，在返回 `layer_input` 前按调用侧传入的 `norm_weight`/`norm_eps`
执行 native RMSNorm。这样 DLC 的 native/加速路径与原有 HIP fallback 的 norm
边界一致；未传 norm 参数时保持原行为。当前只做 PyTorch native 组合，没有修改
PyTorch DLC C++ 算子。

归一化位置是 mHC pre 加权汇总 `layer_input` 之后；`MHCPostOp` 仍只执行残差混合，
下一次 pre（或 fused post-pre）再使用对应的 norm weight。当前已完成源码语法和补丁
清单检查，DLC 设备上的数值对照待后续运行环境可用后验证。

### 处理建议与验收标准

优先增加 DLC OOT 实现：调用 DLC mHC 原语后，显式应用与 GLM5 公式一致的 RMSNorm；或扩展 DLC 原语接口，使 norm 在同一 kernel 中完成。需要覆盖：

- `hc_mult=4`、`sinkhorn_repeat=20`；
- attention 与 FFN 两种 norm weight；
- pre、post、fused post-pre 三种调用形式；
- BF16 输入、FP32 mHC 参数。

验收时将 DLC 路径与 PyTorch reference 的 `layer_input`、`post_mix`、`comb_mix` 逐项比较，并单独比较应用 RMSNorm 前后的结果。

## 5. AWQ MoE 的 SwiGLU clamp 被硬编码为 7（代码已处理，设备验证待做）

### 代码证据

[`vllm_cl/ops/awq_marlin.py`](/root/workspace/vllm-cl/vllm_cl/ops/awq_marlin.py:365) 的 `DLCAWQMarlinMoEMethod.apply_monolithic` 调用 DLC fused MoE 时原先写死：

```python
swiglu_limit=7.0
```

目标模型配置为 `swiglu_limit=10.0`，而 vLLM GLM5 模型构造已经从 `config.swiglu_limit` 读取 10.0。

### 影响

42 个 AWQ MoE 层会使用错误的激活截断范围。服务即使成功启动，输出也不是目标模型计算图的结果。

### 已完成的处理

`apply_monolithic` 现在从 `RoutedExperts.swiglu_limit` 读取配置，并将值原样传给
`torch.dlc_vllm_fused_moe_w4a16_awq_col_major`。因此 GLM-5.3-Flash 的 10.0 会生效，
未设置 clamp 的模型仍传递 `None`，由 fused op 保持无截断语义。同时将算子返回值保存后再
返回，修复该调用点原有的未定义局部变量问题。

DLC kernel 将带 clamp 的 SiLU 实现命名为 `swiglustep`，而 GLM-5 的模型配置仍为
`silu` 并单独携带 limit；caller 在 limit 非空时将 activation 映射为
`swiglustep`，在 limit 为空时保留无 clamp 的 `silu` 路径。这样 kernel 才会实际使用
10.0，而不是只接收一个在 `silu` 分支中被忽略的标量。CPU native reference 也已调整为
先 clamp gate、再执行 SiLU，与 vLLM `SiluAndMulWithClamp` 的数学顺序一致。

当前调用的关键参数为：

```python
fused_output = torch.dlc_vllm_fused_moe_w4a16_awq_col_major(...,
    swiglu_limit=getattr(layer, "swiglu_limit", None),
)
return fused_output
```

新增 [`tests/test_awq_marlin_swiglu.py`](/root/workspace/vllm-cl/tests/test_awq_marlin_swiglu.py)
作为不依赖完整 vLLM/DLC 导入的 CPU 契约测试，覆盖 10.0/`None` 参数和 activation mode
透传，以及 native reference 中 clamp=7 与 clamp=10 的可观察差异和 gate 先 clamp 再
SiLU 的顺序。当前已通过语法检查和 4 项隔离测试。

### 处理建议与验收标准

- 由 MoE layer/config 传递实际 `swiglu_limit`，不要在 vLLM-CL op wrapper 中写常量。
- 保持 fused kernel 的 `None` 语义：没有 clamp 的模型仍走原有路径。
- 用相同 hidden states、router top-k 和反量化权重，对比 clamp=7 与 clamp=10，确认 GLM5 使用的是 10；再做整层 reference 对比。

设备级验证仍待完成：在 DLC 上加载真实 AWQ 权重，确认 42 个 routed MoE 层的配置均为
10.0，比较 fused op 与 native reference 的整层输出，并检查未设置 clamp 的模型路径。

### Router 精度阶段决策（2026-09-15）

GLM-5.3-Flash 配置要求 `moe_router_dtype=float32`。当前 GLM5 构造 `GateLinear`
时传入 `out_dtype=torch.float32`，但这不能保证后续 routing 始终保持 FP32：
[`DLCAWQMarlinMoEMethod.apply_monolithic`](/root/workspace/vllm-cl/vllm_cl/ops/awq_marlin.py:330)
在调用 fused MoE 前显式把 `router_logits` 转为 BF16，底层
[`dlc_vllm_fused_moe_w4a16_awq_col_major_impl`](/root/workspace/pytorch/aten/src/ATen/native/dlc/FusedOps.cpp:3174)
也要求 `gating_output.dtype() == at::kBFloat16`。`DLCGateLinear.set_out_dtype()`
虽然另有强制 BF16 的实现，但不能据此认定 GLM5 已在构造时设置 dtype 的路径必然调用该方法。

第一阶段以 native 组合算子完成服务启动和少量请求测试，**暂时接受向 AWQ fused MoE
传入 BF16 routing logits，不把 FP32 router 列为首次启动前的代码阻断**，本轮不修改算子接口。
BF16 舍入可能改变临界 expert 的 Top-8 选择和路由权重；与 FP32 reference 对比时应记录
此项已接受的精度差异，不能把本阶段结果当作 FP32 router 已完成的证据。

后续将 FP32 router 作为算子接口工作跟踪：扩展 fused MoE 的 FP32 logits 输入及 routing
实现，或提供 native FP32 routing 与 expert 计算分离的调用接口；再验证 Top-8、correction
bias、归一化和 routed scaling 的完整精度语义。

## 6. AWQ Dense 层与 inner model 权重映射（代码已处理，静态验证通过，设备验证待做）

### 确认的根因

GLM5 外层继承的 [`Glm4vForConditionalGeneration.packed_modules_mapping`](/root/workspace/vllm/vllm/model_executor/models/glm4_1v.py:1763)
包含 `"gate_up_proj": ["gate_up_proj"]`，适用于已经融合的 GLM4V checkpoint。
本模型前三层 Dense MLP 的 checkpoint 使用独立的 `gate_proj.weight` 和
`up_proj.weight`，AWQ 排除列表也使用这两个名称。

构造 runtime `gate_up_proj` 时，DLC AWQ 的 `get_quant_method()` 使用
[`is_layer_skipped()`](/root/workspace/vllm/vllm/model_executor/layers/quantization/utils/quant_utils.py:612)
将融合层名展开后匹配排除列表。旧映射无法展开到 gate/up 两个 shard，导致本应保持 BF16
的 `gate_up_proj` 进入量化参数分配。随后 loader 虽然能把 gate/up key 合并成
`gate_up_proj.weight`，模块却只有 `qweight/qzeros/scales`，会在加载时失败。

### 已完成的修复与命名空间核对

插件 [`_install_weight_mappings()`](/root/workspace/vllm-cl/vllm_cl/patch/models/patch_glm5next.py:259)
为 GLM5 外层和 inner causal model 各自设置独立的 packed mapping，在实例化之前生效：

| Runtime 模块 | Checkpoint shard |
| --- | --- |
| `gate_up_proj` | `gate_proj`, `up_proj` |
| `fused_qkv_a_proj` | `q_a_proj`, `kv_a_proj_with_mqa` |
| `wk_weights_proj` | `wk`, `weights_proj` |
| `in_proj_qkvbfg_a` | `q_proj`, `k_proj`, `v_proj`, `b_proj`, `f_a_proj`, `g_a_proj` |

这些名称与 [`Glm5NextModel.load_weights()`](/root/workspace/vllm/vllm/models/glm5next/nvidia/model.py:706)
的 stacked mapping 一致。两类拥有各自的字典和 shard 列表，不修改 GLM4V 父类；重复安装
保持相同结果。量化选择仍使用原 AWQ 排除规则，gate/up 只有部分被排除时仍明确报错。

外层前缀 mapper 本身正确，继续沿用：

```text
model.language_model.* -> language_model.model.*
model.visual.*         -> visual.*
lm_head.*              -> language_model.lm_head.*
```

[`configure_quant_config()`](/root/workspace/vllm/vllm/model_executor/model_loader/utils.py:281)
在模型构造前调用 `apply_vllm_mapper()`，先把排除列表转换到 runtime 命名空间，再设置
packed mapping。inner 初始化更新 packed mapping，但不再次转换前缀。

实际加载时，外层 `AutoWeightsLoader` 转换前缀并递归移除 `language_model.`；inner
收到 `model.layers.*`，再递归移除 `model.` 后交给 `Glm5NextModel.load_weights()`。
因此 inner 没有独立 `hf_to_vllm_mapper` 是正确行为；无需追加二次前缀转换。

### 静态审计结果

新增 [`audit_glm5_weight_mapping.py`](/root/workspace/glm-5.3-flash/audit_glm5_weight_mapping.py)
和报告 [`GLM-5.3-Flash_weight_mapping_audit.json`](/root/workspace/glm-5.3-flash/GLM-5.3-Flash_weight_mapping_audit.json)。
审计只读取 safetensors index 和每个可用分片的 8-byte/header，不加载 tensor payload，
并用 recording parameter 执行当前 `Glm5NextModel.load_weights()` 的映射逻辑。结果如下：

- index 与独立 expected catalog 均为 111,346 keys；无 missing、unexpected 或重复目标 slot。
- 可读取分片的 110,706 keys 中，110,457 个产生参数加载调用；889 个 layer 45 MTP keys
  被 loader 明确跳过。
- 分类计数为 routed AWQ 108,864、Dense BF16 9、KDA BF16 442、MLA/indexer BF16 154、
  shared BF16 126、vision BF16 347，以及 router/mHC FP32 参数。
- 当前目录缺少 `model-00104-of-00120.safetensors` 和
  `model-00105-of-00120.safetensors`，因此对应 640 keys 没有 header；这是 checkpoint
  传输尚未完成造成的输入不完整，未被误报为映射错误。

报告还保留了修复前的量化选择基线：旧的 `gate_up_proj: [gate_up_proj]` 映射使 layer
0/1/2 的融合 Dense 层均判定为未跳过；修复后的
`gate_up_proj: [gate_proj, up_proj]` 要求三层均判定为跳过。这样可在未来修改映射时
直接发现回归。

同时新增 [`test_glm5_awq_weight_mapping.py`](/root/workspace/vllm-cl/tests/test_glm5_awq_weight_mapping.py)，
覆盖父类映射隔离、外层 prefix mapper、量化排除规则、Dense/MoE 形状和 dtype、以及
stacked shard 的唯一目标映射。契约测试与审计脚本均通过。

### 本阶段边界

前三层 Dense、attention、shared experts、router 和 lm_head 应保持 checkpoint 的
非 AWQ 参数类型；第 3–44 层 routed experts 使用 W4A16 AWQ。Layer 45 属于 MTP，
初期不启用 speculative decoding，由主模型现有 loader 显式跳过。

真实 DLC 上的完整模型构造、权重加载、TP 切片、post-load 列布局变换和推理数值仍待验证；
缺失的两个分片补齐后需要重新运行审计。
本节的代码与静态检查不代表整个模型服务已适配完成。

## 7. KPool Indexer 状态链路（问题 1–4 已修复，设备验证待做）

修订日期：2026-09-14。本节中的“head size 576”问题已在第 3 节处理，不再属于阻断点 7。本轮仅修复此前列出的 KPool 问题 1–4；验证使用 CPU 小张量与当前主仓库的 metadata/manager 代码，未启动服务、未加载真实权重。

### 本轮问题 1–4 的处理结果

| 问题 | 修复内容 | 验证重点 |
| --- | --- | --- |
| 1. tail cache 选错 backend | `DLCGlm5NextTailCache` 显式返回 `KpoolTailBackend`，绑定时检查 `[blocks,2,4,head_dim]`。 | backend 选择、pool/tail 不同布局绑定。 |
| 2. 缺失 tail metadata 静默返回全 `-1` | pool/tail builder 均补齐请求边界、原始序列长度、物理 block table、slot mapping、实际 token 数和块大小；真实 forward 缺字段或未绑定 cache 明确报错。只有 `attn_metadata=None` 的 profiling 路径允许空 TopK。 | 缺 metadata/字段、错误绑定、actual token 与输出 padding。 |
| 3. `_tail_lengths` 按 batch 行号保存，无法跟随请求换序和恢复 | 从 `(seq_len-query_len)%4` 推导尾池长度，按 tail 物理页恢复 K/gate，删除 caller 的本地长度数组。新增专用 tail manager，每个请求只占一个稳定物理块，释放时归还。 | A/B 换序后正确完成原尾池；重建 caller 并恢复 cache；物理块复用；长 prefill/decode 不扩容或裁剪 tail。 |
| 4. 忽略实际 slot mapping，pool 物理页与虚拟页单位不一致 | 计算消费两个 builder 的真实 slot mapping；pool slot 按压缩状态计数，tail slot 按原始 token 的 ring offset 计数。固定 pool storage block size，保留上游归并后的物理表，并在写入前检查映射一致性。 | 256-token 物理页 / 128-token 虚拟页、跨页、`3→4→5`、`255→256→257`、padding 与错误 slot 拒绝。 |

专用 manager 同时解决了本轮复核发现的两个实际阻断：现有通用工厂向 `KpoolTailSpec.max_admission_blocks_per_request` 传入旧参数名会触发 `TypeError`；回退的 `SlidingWindowManager` 会随序列增长分配并裁剪 tail 块，违反单块环形缓存协议。插件仅接管 `KpoolTailSpec` 的工厂和注册，普通 cache manager 保持原路径。

V1/V2 worker 也已接入 [`kpool_worker_slots.py`](/root/workspace/vllm-cl/vllm_cl/kpool_worker_slots.py)。通用 DLC slot kernel 用 `position//block_size` 查询递增的表列，不适用于只占一页的 tail；现在 tail 分支在调用通用 kernel 前计算 `block_table[request,0]*4+position%4`。V2 同时覆盖组合的 prepare 路径和独立 compute 路径，保留其他 cache group 的原有 dispatch。

主要实现位于 [`glm5_indexer.py`](/root/workspace/vllm-cl/vllm_cl/patch/models/glm5_indexer.py)、[`patch_mla_indexer.py`](/root/workspace/vllm-cl/vllm_cl/patch/patch_mla_indexer.py)、[`kpool_indexer.py`](/root/workspace/vllm-cl/vllm_cl/patch/models/kpool_indexer.py) 和 [`patch_kpool_cache_manager.py`](/root/workspace/vllm-cl/vllm_cl/patch/patch_kpool_cache_manager.py)。计算仍采用插件内 PyTorch native；未引入 NVIDIA attention 执行或 Triton 计算。

### 本轮验证结果

按当前文件统计，11 个主隔离测试文件在修复 tail 清理后实际通过 **87/87**；此前 `test_bf16_gate_storage_is_chunking_invariant` 暴露的分块一致性缺陷已在 `NativeKPoolPagedCache.append` 中清理已完成 tail 的两块存储并复测通过。另有 `test_kpool_weight_layout.py` 4 项和 `test_kpool_speculative.py` 4 项在注入最小 pytest shim 后通过。原环境未安装 pytest，因此这两项不是标准 pytest 运行结果。本轮实现文件的 Python 语法检查、patch import manifest 的 YAML/路径
检查及 `git diff --check` 均通过。vLLM 主仓库与 PyTorch 仓库未修改。

| 测试文件（位于 `vllm-cl/tests/`） | 数量 | 覆盖范围 |
| --- | ---: | --- |
| `test_kpool_indexer_native.py` | 14 | 压缩/打分 reference、跨 chunk、请求重排、恢复和 cache 写入校验。 |
| `test_kpool_slot_mapping_native.py` | 9 | 压缩 slot、跨页、空请求与 padding。 |
| `test_kpool_metadata_native.py` | 9 | 实际 upstream builder 构造/归并流程、tail metadata、非 GLM5 隔离。 |
| `test_glm5_indexer_integration.py` | 12 | 实际插件 caller、worker→builder→caller→native cache，以及真实 metadata 缺失报错。 |
| `test_kpool_cache_manager.py` | 9 | 实际 upstream 工厂/registry/manager 源码，单页分配、释放、重用和普通 manager 隔离。 |
| `test_kpool_worker_slots.py` | 7 | V1/V2 tail 分支、位置 4/128/256/4097、持久请求索引与普通 kernel dispatch。 |
| `test_kpool_cache_layout.py` | 2 | pool/tail 布局与 DLC 物理页字节数。 |
| `test_glm5next_dlc_contract.py` | 9 | 第 3 阻断点的 MLA caller、512 维 native attention 回归。 |
| `test_glm5_awq_weight_mapping.py` | 7 | 第 6 阻断点的 prefix/packed mapping、量化排除和参数契约。 |
| `test_awq_marlin_swiglu.py` | 4 | 第 5 阻断点的 clamp 参数、activation mode 和 native 数学顺序。 |
| `test_kda_native.py` | 5 | 第 2 阻断点的 native recurrence、分页状态、卷积、RMSNorm 公式 smoke test 与 speculative 保护。 |
| `test_kpool_weight_layout.py` | 4 | KPool 权重布局转换契约（仅 pytest shim 补测）。 |
| `test_kpool_speculative.py` | 4 | KPool speculative journal 契约（仅 pytest shim 补测）。 |

测试通过 importlib/AST 隔离环境依赖，执行当前实现源码和 CPU PyTorch；容器缺少 pytest，完整插件 import 还受到 torchaudio 的 `libcudart.so.13` 依赖限制。这些测试不覆盖完整服务导入、DLC 设备执行或真实权重精度；KDA 的 RMSNorm 项为等价公式 smoke test，多 token 项为状态/有限值检查，尚未完成端到端 caller 与 golden 数值对照。复现命令：

```bash
cd /root/workspace/vllm-cl
/usr/bin/python3.10 - <<'PY'
import importlib.util
import unittest
suite = unittest.TestSuite()
names = [
    'kpool_indexer_native', 'kpool_slot_mapping_native', 'kpool_metadata_native',
    'glm5_indexer_integration', 'kpool_cache_manager', 'kpool_worker_slots',
    'kpool_cache_layout', 'glm5next_dlc_contract', 'glm5_awq_weight_mapping',
    'awq_marlin_swiglu', 'kda_native',
]
for name in names:
    spec = importlib.util.spec_from_file_location(name, f'tests/test_{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    suite.addTests(unittest.FunctionTestCase(fn) for key, fn in vars(module).items()
                   if key.startswith('test_') and callable(fn))
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

### 本快照改动清单

本分支包含本轮 GLM5 适配涉及的全部 vLLM-CL 源码和回归材料：GLM5 MLA/indexer/model binding（`vllm_cl/patch/models/patch_glm5next.py`、`glm5_indexer.py`、`kpool_indexer.py`、`kpool_slot_mapping.py`）、KPool cache layout/manager/worker slot 与 speculative journal、MLA/KV-cache/block-table/worker V1/V2 接线、mHC RMSNorm 补丁、AWQ SwiGLU 精度与参数修复、KDA native 初稿（`patch_kda.py`），以及 `patch_import_manifest.yaml`、启动脚本和本目录中的权重映射审计材料。`vllm_cl/_version.py` 的变更是 setuptools-scm 生成版本漂移，不代表模型逻辑。

### 问题 5–7 状态（历史记录）

下表中的 5–7 是 KPool/MTP/eager 子问题编号；本文件上方的第 6 节是独立的
AWQ Dense 权重映射阻断点，已在本轮处理。

| 后续问题 | 当前边界 |
| --- | --- |
| 5. 精度策略（本轮已处理） | gate 在投影后立即按原模型协议转 BF16，并与 K 一起写入 BF16 tail；`compress_kpool()` 入口统一将 K/gate/APE 转 FP32，FP32 完成 logits、softmax 和池累加，结果按 cache dtype 回写 BF16。head 权重在 post-load hook 中还原 DLC 物理列布局后切片并缓存为 FP32，forward 使用 FP32 GEMM。 |
| 6. speculative rollback（初期暂缓） | 初期适配不启用 MTP/speculative decoding，普通 prefill/decode 路径不受影响。V1/V2 已保留 native transaction 接线和参考回滚框架，但多层 KPool 的 request-granular journal、部分接受重放、异步换序和真实设备回归暂不作为本阶段关闭条件；后续性能优化、启用 MTP 时再集中处理。 |
| 7. graph / eager 配置（初期适配已关闭） | 启动脚本显式传入 `--enforce-eager`，源码只校验配置并拒绝误入 graph，不修改默认配置；融合算子替换 native 后再重新评估 graph/capture。 |

问题 5 的数值协议已固定为分层策略。模型语义遵循 HF 的公式和顺序：池内对
`gate + APE` 做 softmax 加权 K，索引分数执行逐 head 点积、ReLU、head 权重汇总，
再做 pool TopK 并展开 token。gate、raw K 和 tail cache 使用 BF16；压缩入口统一
转换 K/gate/APE 为 FP32，FP32 完成 logits、softmax 和累加，pooled K 回写 BF16。
head 权重采用 NVIDIA 的 FP32 GEMM 精度策略，但不在 forward 切片：模型加载完成后，
post-load hook 先逆 DLC 列布局，再切出逻辑 head 行并缓存 FP32 投影矩阵。NVIDIA
实现没有模型语义错误，HF 是公式 golden，NVIDIA 提供中间精度和存储协议参考。

当前 native 实现仍通过一次 `wk_weights_proj(hidden_states)` 计算融合输出，随后只
使用其中的 K 部分；为保证 head 权重 FP32 GEMM，head 部分会再次使用缓存矩阵计算，
因此 fused 输出的后半段存在暂时的重复计算。这是 native 阶段的性能代价，不影响
数学正确性；后续 DLC 融合算子应支持 K/head 双输出或按输出区间计算，以消除该浪费。

初期 native 验证阶段不在源码中强制修改 eager 配置，改由启动脚本显式传入参数：[`start_glm5_flash_vllm_cl.sh`](/root/workspace/glm-5.3-flash/start_glm5_flash_vllm_cl.sh)。脚本默认使用模型路径 `/data/models/GLM-5.3-Flash-Mix`、`--block-size 256` 和 `--enforce-eager`，可通过 `GLM5_MODEL`、`VLLM_ROOT`、`VLLM_BIN`、`TENSOR_PARALLEL_SIZE`、`PORT` 环境变量覆盖；额外参数会原样传给 `vllm serve`。插件 caller 只做配置生效校验，不主动设置 eager；forward 还会拒绝非 `NONE` 的 graph runtime 或正在 capture 的 DLC stream。对应测试覆盖配置丢失和误入 graph 的错误路径。本轮只做静态检查，未启动服务。融合算子替换 native 后，再恢复 graph/capture 的专项评估。

**2026-09-17 更新：V1/V2 拒绝 KDA `MambaSpec` 的 cache 初始化代码缺口已补齐，待真实设备验证。** [`mamba_cache.py`](/root/workspace/vllm-cl/vllm_cl/worker/mamba_cache.py) 为每个 KDA 层分配独立、清零的 `int8 [num_blocks,1,1,page_size_bytes]` 原始字节页；V1 [`dlc_model_runner.py`](/root/workspace/vllm-cl/vllm_cl/worker/dlc_model_runner.py) 和 V2 [`dlc_model_runner_v2.py`](/root/workspace/vllm-cl/vllm_cl/worker/dlc_model_runner_v2.py) 在 attention 物理布局分支前处理 `MambaSpec`，使用包含 `page_size_padded` 的完整页宽与 scheduler 的 `num_blocks`，再经现有 `bind_kv_cache()` 交给 `MambaBase` 拆成 BF16 卷积状态和 FP32 递归状态。CPU 测试 [`test_dlc_mamba_cache.py`](/root/workspace/vllm-cl/tests/test_dlc_mamba_cache.py) 覆盖混合 attention/KDA 分组、padding 后的 block stride、共享/独立层、state 共享存储及 V1/V2 绑定。原先在进入 KDA forward 前直接抛出 `NotImplementedError` 的路径现已移除；仍需实机检查分配、清零、prefill→decode 状态连续性与多请求正确性。

其他设备级验收仍包括：

- 权重映射、DLC col-major 还原、TP/共享 buffer 生命周期以及真实 checkpoint 加载尚未在设备上验证；
- pool 压缩、逐 head ReLU 打分、TopK 展开和 sparse MLA 输出需要比较 DLC 设备结果与 reference；
- pooled cache 的 `[blocks,64,128]`、tail 的 `[blocks,2,4,128]` 布局及 DLC BF16 物理页字节数已有 CPU 检查，真实分配/bind 仍待验证；
- FP8/scale cache、DCP/PCP、PD tail 传递仍未适配或验证。

当前 factory 已指向 native caller，但只接受 DCP=PCP=1、`index_kpool=4`。恢复发生在未完成的 pool 中时，必须同时恢复该请求的 raw tail；新增 manager 只负责分配接收页，不实现 PD connector 的内容传输。在真实 cache、metadata 和 tail 状态的设备验证完成前，不应把服务视为适配完成。native speculative transaction 当前采用完整 cache 快照并重放 committed 前缀，仅保留为后续实验参考；初期服务不启用 MTP/speculative decoding。生产融合算子应改为按 touched page 的 journal/COW 方案，MTP 相关问题在后续性能优化阶段单独处理。

### GLM5 需求

GLM5 KPool-DSA 的池级分数应为：

```text
head_score[t,h,p] = ReLU(dot(q[t,h], k_pool[p]) / sqrt(128))
score[t,p] = sum_h (w[t,h] / sqrt(32)) * head_score[t,h,p]
```

即先逐 head 点积，再做 ReLU，再按 head 权重汇总；之后对 pool 做 TopK=512，并展开成最多 2048 个 token，加当前 tail token。

### 当前 DLC 参考实现

[`pytorch/aten/src/ATen/native/dlc/muti_query_attention.cxx`](/root/workspace/pytorch/aten/src/ATen/native/dlc/muti_query_attention.cxx:37) 的 CPU reference 是：

```text
q_weighted = (query * weights).sum(head)
logits = q_weighted @ key.T
```

它没有逐 head dot，也没有 ReLU。paged 版本同样先做 weighted query 再矩阵乘。这个公式一般不等价于 GLM5 KPool 分数。

vLLM GLM5 indexer 在 [`vllm/vllm/models/glm5next/nvidia/attention.py`](/root/workspace/vllm/vllm/models/glm5next/nvidia/attention.py:335) 生成 per-head query/weights 后，最终依赖 MQA logits 算子；因此不能仅凭输入 shape 判断语义正确。

### 相关 cache/backend 约束

- `index_kpool=4`，`--block-size 256` 时池存储块为 64，满足按 32/64 进行虚拟 page 切分的约束。
- vLLM-CL 的 sparse MLA backend 已声明 GLM5 所需的 512；这里仍需验证 KPool indexer cache 与 MLA cache 的两套物理布局不会混用。
- `index_topk=2048` 是 token 数；池级 TopK 实际是 512，再展开为 2048 token，尾池最多再加 3 个 token，buffer 可能按 2176 对齐。

### 影响

- TopK 选择错误会改变 sparse MLA 的可见上下文，属于模型语义错误。
- 仅测试输出 shape 或无负分的简单输入无法证明正确；ReLU 和 head 加权的顺序必须单独验证。

### 处理建议与验收标准

本轮已新增插件内 [`kpool_indexer.py`](/root/workspace/vllm-cl/vllm_cl/patch/models/kpool_indexer.py) 作为纯 PyTorch native/reference，包含 pool 压缩、逐 head ReLU 打分、pool-to-token 展开、paged pool cache、tail ring 及 prefill/decode 状态更新；[`kpool_slot_mapping.py`](/root/workspace/vllm-cl/vllm_cl/patch/models/kpool_slot_mapping.py) 提供 compressed slot mapping；[`glm5_indexer.py`](/root/workspace/vllm-cl/vllm_cl/patch/models/glm5_indexer.py) 提供插件 caller 和 cache spec；[`kpool_cache_layout.py`](/root/workspace/vllm-cl/vllm_cl/kpool_cache_layout.py) 统一物理布局。对应 CPU 检查位于 [`test_kpool_indexer_native.py`](/root/workspace/vllm-cl/tests/test_kpool_indexer_native.py)、[`test_kpool_slot_mapping_native.py`](/root/workspace/vllm-cl/tests/test_kpool_slot_mapping_native.py)、[`test_kpool_cache_layout.py`](/root/workspace/vllm-cl/tests/test_kpool_cache_layout.py)；其中 `test_bf16_gate_storage_is_chunking_invariant` 覆盖 BF16 gate/tail 的整段与分块一致性。

1. 用真实 DLC 设备和 checkpoint 验证权重加载、cache bind、prefill→decode 3→4→5 和多请求 page table。
2. 确认 `torch.dlc_multi_query_attention`/paged 版本的硬件真实语义；native caller 当前绕过 weighted-query MQA，直接执行逐 head dot→ReLU→head 汇总。
3. 初期覆盖完整池、尾池、空槽 `-1`、TopK 边界和 decode page table；MTP `next_n>1`
   留到后续性能优化阶段。
4. 用相同输入比较 DLC 与 golden 的 `pool_ids` 完全一致，再比较 sparse MLA 输出。

## 建议处理顺序

为减少相互干扰，建议按以下顺序逐项解决：

1. 先用插件 GLM5 MLA 类适配当前自研 wrapper，并实现 512 维 native sparse attention（详见第 3 节）。
2. 建立无 Triton 的 KDA BF16 prefill/decode reference 路径。
3. 修复 mHC RMSNorm 语义。
4. 修复 AWQ MoE 的动态 `swiglu_limit`。
5. 统一外层/inner loader 命名空间，完成 Dense/AWQ 参数覆盖检查。
6. 建立 KPool golden，确认 DLC MQA 算子语义后再接入 sparse MLA。

每个阻断点完成后，都应先做对应的单元/reference 检查，再进入下一项；在这些静态问题解决前，不应把服务启动成功作为适配完成的判断依据。
