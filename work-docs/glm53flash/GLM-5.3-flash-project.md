# GLM-5.3-Flash 推理适配项目总结

> 文档性质：项目总结（内部归档，正文按对外可披露口径撰写）。  
> 事实截止：2026-11-02；整理日期：2026-11-02。  
> 项目状态：已由云服务团队正式生产上线；AIME26 端到端 30/30，与官方表现指标一致。  
> 披露口径：芯片统一写“自研 AI 加速芯片”，不写内部代号、内部仓库名和内部算子编号。

## 0. 结论摘要

在 8 卡自研 AI 加速芯片上，独立完成 GLM-5.3-Flash 的 vLLM 插件侧适配、性能优化、NaN 排障和发布收敛。项目把一套依赖 Triton、TileLang 和 `torch.compile` 的上游实现，改造成可在自研芯片上运行的 TP=8、BF16 激活、AWQ 4-bit、`FULL_DECODE_ONLY` decode graph 的文本推理路径。

最终结果可以概括为五点：

- **已生产上线**：云服务团队正式生产上线；AIME26 端到端 30/30 全对，与官方表现指标一致。
- **能跑通**：服务启动、健康检查、单请求、8 并发、chunked prefill、长上下文和 graph/eager 对照均完成验证；graph 与 eager 在相同输入下逐字一致。
- **性能可用**：首次真机可用前，prefill/decode 单步约 179 s/120 s；状态拆池后降到约 5.7 s/2.6 s。后续 profile 驱动优化把干净 decode 做到约 96.9 ms/token，decode graph 单请求 gap 从约 110 ms 降到约 68 ms；开启 graph replay cache 后再降约 9%–10%。
- **质量可控**：插件定向回归 `161 passed, 10 subtests passed`；11.1 万个权重 key 完成静态映射审计；全 `!`/NaN 故障定位并修复两个独立根因，形成可复用的排查 SOP。
- **边界清楚**：当前上线版本是文本、BF16 激活、V1 runner、decode graph 的发布版本；MTP、多模态、FP8、V2 runner、计算/通信重叠不在本次范围内。KPool select/top-k 的长上下文历史退化已定位到该融合算子，优化后的长上下文预估值见 6.3 节。

## 1. 项目概览

| 项目 | 内容 |
|---|---|
| 时间 | 2026-09-01 至 2026-10-27，约 8 周；2026-11-02 更新生产上线状态 |
| 角色 | AI 框架开发工程师；独立负责本模型在 vLLM 插件侧的全部适配开发，并对接算子、PyTorch 后端、运行时和发布验证 |
| 模型 | GLM-5.3-Flash-Mix；约 321B 总参数、18B 激活参数；45 层混合注意力；模型声明上下文 1,048,576 tokens |
| 部署形态 | 8 卡自研 AI 加速芯片，TP=8，AWQ 4-bit 权重、BF16 激活，`block_size=256`，文本推理 |
| 软件栈 | vLLM 0.30.1rc0、自研 vLLM 硬件插件、PyTorch 2.5.0 芯片后端、自研 AOT 算子库；Python + C++ |
| 交付状态 | 已由云服务团队正式生产上线；AIME26 端到端 30/30，与官方表现指标一致 |

项目开始时的主要判断是：**这不是“注册一个模型、补几个 op”就能完成的工作**。GLM-5.3-Flash 同时引入了 KDA 线性注意力、KPool-DSA 稀疏注意力、NoPE 稀疏 MLA、四路 mHC 残差、AWQ MoE 和多种有状态缓存；上游实现又深度依赖 NVIDIA 生态。适配工作必须同时处理模型语义、缓存生命周期、芯片 ABI、布局/对齐、算子接口和 host/device 流水线。

## 2. 模型结构与平台约束

### 2.1 模型结构账本

| 维度 | 配置 | 对适配的影响 |
|---|---|---|
| 层数与注意力 | 45 层，`[KDA×3 + 稀疏 MLA]×11 + KDA` | 34 层线性注意力与 11 层稀疏注意力交替，不能按单一 Transformer 处理 |
| KDA | 64 heads、head dim 128、短卷积宽度 4 | 每层同时维护卷积状态和递归状态；prefill 与 decode 必须分开实现 |
| KPool indexer | 32 heads、dim 128、`index_kpool=4`、`index_topk=2048` | 每 4 个 token 压缩成一个池，池级 TopK 后再展开成 token；还有尾池环形缓存 |
| 稀疏 MLA | latent 512、`qk_rope_head_dim=0` | 原有 576 维路径不可直接复用，需要 512 维 NoPE 路径；TopK 输出按 2176 对齐 |
| mHC | 每层 4 路 residual stream、20 轮 Sinkhorn | 残差混合和 RMSNorm 边界容易出错，且小算子很多 |
| MoE | 后 42 层 MoE；288 routed experts、Top-8、1 个共享 expert | 需要 AWQ 布局、动态 `swiglu_limit=10` 和主机侧反量化兜底 |
| 权重 | 4-bit AWQ/group-size 128，120 个分片，约 178.81 GiB | TP=8 每 rank 约 21.55 GiB；权重重排和 col-major 变换必须准确 |
| 其他结构 | 1 层 MTP、视觉塔 | 存在于 checkpoint，但不在本次发布路径；不能把“模型里有”误写成“已支持” |

KDA 的状态量也需要单独核算：单请求全局约 `140.8 MiB`，TP=8 后每 rank 约 `17.6 MiB`。这类状态不是普通 KV cache，不能简单套用 MLA 的页布局。

### 2.2 平台约束

| 约束 | 具体表现 | 设计后果 |
|---|---|---|
| 无 Triton/TileLang | 上游 KDA、indexer、mHC 等路径依赖 NVIDIA kernel 或 `torch.compile` | 必须建立插件自有 native 路径，再逐步替换成芯片融合算子 |
| 目标 ABI 指针宽度不同 | 芯片编译目标下 `unsigned long`/指针为 32 位 | 所有地址单位换算都要显式使用 64 位；否则超过 16 GiB 后静默回绕 |
| 物理行与对齐严格 | 部分算子要求一条逻辑行对应一条 512 B 物理行 | view 操作数往往必须物化，或扩展算子 ABI 吃 `(base, offset, pitch)` |
| 非连续 view 代价极高 | 非连续张量会掉出 DMA 路径，退化为 host 逐元素搬运 | 打包状态页会把所有搬运算子拖慢数千倍，必须从布局上消除 |
| 固定 `block_size=256` | 上游混合 KV cache 的共享页假设不成立 | 需要 GLM 专用的 MLA/KPool/KDA 分组和容量核算 |
| 无计算/通信重叠 | 集合通信与计算在同一 stream 上串行 | host 下发次数、屏障等待和 D2H 阻塞会直接进入 decode 关键路径 |
| V2 runner 不可用 | V2 需要 Triton，本机没有可用 driver | 发布路径固定为 V1 runner；Graph 修复也按 V1 验证 |

## 3. 职责与协作边界

### 3.1 本人负责

- 插件内 GLM-5.3-Flash 的模型注册、权重映射、模型/indexer/sparse MLA 绑定和 native 执行路径。
- 混合 KV cache 分组、KDA/Mamba 状态池、容量核算、block table 和状态迁移。
- KDA、KPool、mHC、NoPE 稀疏 MLA 的插件侧调用契约、融合算子接入和回退开关。
- 真机联调、启动/推理阻断排查、端到端 profile 和插件侧性能优化。
- 融合算子需求梳理、已有算子覆盖核查、交付算子验收和“是否真的走到 kernel”的测试设计。
- Graph 捕获适配、KDA padding 边界修复、发布代码清理、定向回归和交付文档。

### 3.2 共同完成或推动

- 与算子组共同完成 KDA 系列算子的缺陷闭环：本人负责定位、复现、验收和接入，算子组负责 kernel 实现与性能优化。
- 向 PyTorch 后端团队提交 strided copy 静默错误的最小复现和影响面分析，推动后端修复。
- 与运行时/通信团队一起拆分 allreduce 的“等同步”和“通信本体”，为后续优化提供归因数据。

### 3.3 不由本人负责

- 融合算子的 kernel 实现和 kernel 级性能优化，包括 KDA、KPool、mHC、稀疏 MLA 等 kernel。
- 芯片 runtime、驱动、集合通信库和编译器后端的实现。
- 插件平台框架、以往机型适配和已有算子基础设施；这些是本项目开展前的团队基础。

## 4. 适配全过程

| 时间 | 阶段 | 主要工作与结果 |
|---|---|---|
| 09-01 至 09-16 | 结构研究、静态适配 | 对照官方配置、HF/Transformers 参考实现和上游 vLLM PR，建立模型结构账本、权重映射和缓存/算子缺口矩阵；确定“先 BF16 文本、无 MTP、TP-only 打通，再做融合与图”的路线 |
| 09-17 至 09-21 | 首次真机联调 | 对齐 vLLM 0.30.1rc0 基座；修掉视觉塔 Triton RMSNorm、AWQ `qzeros` 布局、两轮 KV cache OOM、KV manager 返回契约、`searchsorted`/`uint64` 缺失等问题；服务可启动并接受请求 |
| 09-21 至 09-22 | 算子覆盖核查 | 对 18 项候选融合需求做真机核查，区分“新增/扩展/修复/复用”，形成 12 条首轮提交；同步修正 mHC RMSNorm、SwiGLU clamp、Dense/AWQ 权重映射等语义问题 |
| 09-28 | 最大性能拐点 | 通过 py-spy、隔离微基准和模型内 profile 定位“打包状态页 → 非连续 view → host 逐元素搬运”的布局税；实现 KDA 状态拆池，prefill/decode 单步约 179 s/120 s 降到约 5.7 s/2.6 s |
| 09-29 至 10-10 | 融合算子与流水线优化 | 接入 KDA 短卷积、chunk prefill、recurrent decode、gated RMSNorm、state gather/scatter；接入 512-NoPE 稀疏 MLA；合并 metadata H2D、去掉长阻塞 D2H；clean decode 基线约 233 ms/token |
| 10-13 至 10-21 | KPool、Graph 与发布路径 | 完成 KPool 更新/压缩、select/top-k/expand、slot mapping 三条融合链；补齐 KDA graph padding、持久 slot buffer 和 `FULL_DECODE_ONLY` 捕获 |
| 10-21 至 10-22 | NaN 与发布验证 | 定位并修复 KDA 16 GiB 地址溢出、sparse MLA null block warmup 污染两个独立根因；完成 1/2/8/16/24/32 并发 graph 矩阵、graph/eager 对照和发布前功能/性能验证 |
| 10-23 | 代码收敛 | 删除 native/fallback、CPU 对比分发和调试探针，统一生产路径；定向回归 `161 passed, 10 subtests passed`；发布 smoke 通过 |
| 10-26 至 10-27 | 最终复测与发布 | 开启 graph replay cache 复测，单并发 decode gap 再降约 9%–10%；修复 block numel 校验；完成发布基线和部署文档 |
| 11-02 | 生产上线与精度验证 | 云服务团队正式生产上线；AIME26 端到端 30/30 全对，与官方表现指标一致；KPool 长上下文瓶颈定位到 select/top-k 融合算子，优化后预估值见 6.3 节 |

## 5. 关键方案

### 5.1 模型接入与权重映射

- 在插件中建立 GLM5Next 的模型/indexer/稀疏 MLA/缓存绑定，不直接复制 NVIDIA 实现，也不在 import 阶段依赖 Triton。
- 补齐四组关键打包映射：`gate_up`、`fused_qkv_a`、`wk_weights`、`in_proj_qkvbfg_a`；外层和 inner model 使用独立命名空间，避免继承 GLM4V 父类映射后误量化 Dense 层。
- 编写只读 safetensors header/index 的静态审计脚本，确认 111,346 个权重 key 无 missing/unexpected/重复目标；其中 889 个第 45 层 MTP key 因本次不启用 MTP 被明确跳过。
- 对 KDA、KPool、mHC、稀疏 MLA 先建立 CPU/FP32 reference 和 native 实现，再逐个替换为芯片融合算子；每个阶段保留可观测的回退路径，发布前再统一删除。

### 5.2 混合 KV cache：从两次 OOM 到可计算容量

第一次 OOM 的根因不是“显存不够”，而是**分组、核算、分配三者不一致**：

- 上游把 34 个 KDA 层合并成一个组，并假设 KDA 状态页可以装进 MLA 页；但芯片侧实际按层独立分配，且 KDA 真实页约 0.518 MiB，大于 MLA 的 0.25 MiB。
- 结果 `num_blocks` 被高估约 6.5 倍，启动日志只有 45,409 tokens 的 KV cache，并出现 674 MiB 分配失败。
- 修复方式是在插件入口把 KDA 拆成每层一个独立 spec，按真实页大小计费，形成“1 组 full MLA + 1 组 KPool index + 1 组 KPool tail + 34 个 KDA 独立组”。

第二次 OOM 来自平台层为共享页布局做的 Mamba 页 padding：

- KDA 真实页为 `542,720 B`，被补到 `786,432 B`，约浪费 5.7 GiB。
- 确认 `bind_kv_cache` 只访问真实页后，跳过该 padding，使核算和实际分配统一使用真实页。
- 修复后 34 层 KDA 全部分配成功，KV cache 提升到 148,762 tokens（0.85 显存利用率）；后续生产配置在 0.80 下为 1,411 blocks。

当前单请求 block 预算可写为：

```text
blocks_per_request(L) = 69 + 2 × ceil(L / 256)
```

其中 69 来自 34 层 KDA 各 2 个状态 block，加 1 个 KPool tail block。这个公式把“模型结构”直接变成了可验收的容量约束。

### 5.3 状态拆池：从“布局税”到最大性能拐点

这是整个项目最有代表性的性能问题。

**现象**：服务能启动，但 prefill 单步从数十秒恶化到约 179 s，decode 单步约 120 s；芯片上几乎看不到计算，8 个 worker 却各占约 200% CPU。

**定位**：旧布局把每层的 conv state 和 recurrent state 打包在同一个 `542,720 B` 字节页里，绑定时按字节偏移切出两个 view：

- conv state 的 `stride(0)=271,360`，自身紧凑布局应为 `9,216`；
- recurrent state 的 `stride(0)=135,680`，自身紧凑布局应为 `131,072`。

两个 view 都非连续。芯片后端对这类张量会掉出 DMA 路径，退化为 host 逐元素搬运，且耗时与整个 storage 大小相关，而不是与实际动了几页相关。隔离微基准显示：

| 操作（只动 8 页 / 4 MiB） | 连续池 | padded 页内 view | 放大 |
|---|---:|---:|---:|
| `index_select` + clone | 0.181 ms | 579.7 ms | 3,196× |
| `index_copy_` | 1.544 ms | 1,873.0 ms | 1,213× |
| KDA conv prefill | 0.166 ms | 1,396.1 ms | 8,400× |
| KDA conv update | 0.259 ms | 1,339.3 ms | 5,200× |

模型内 profile 也吻合：prefill 一步 179 s 中，scatter、conv prefill、gather 合计约 96%；decode 一步 120 s 中，conv update 与 recurrent 合计约 96%。所以问题不是某个 kernel 算得慢，而是**布局让所有搬运算子一起退化**。

**修复**：新增 `MambaStatePool`，把 conv state 和 recurrent state 拆成两个独立连续池；总字节数、block id 语义和 512 B 对齐保持不变，并提供回退开关。同步适配 KV 清零、runner 绑定、状态迁移和容器型 cache 条目。

**结果**：

| 指标 | 拆池前 | 拆池后 | 变化 |
|---|---:|---:|---:|
| prefill 单步 host | 179,434 ms | 5,703 ms | 约 31× |
| decode 单步 host | 119,962 ms | 2,579 ms | 约 46× |
| KDA conv prefill | 1,941 ms/call | 0.40 ms/call | 约 4,800× |
| KDA conv update | 1,835 ms/call | 0.19 ms/call | 约 9,700× |
| state gather | 729 ms/call | 44 ms/call | 约 17× |
| state scatter | 2,444 ms/call | 36 ms/call | 约 68× |

这里的 179 s → 5.7 s、120 s → 2.6 s 是**单步 host 耗时**，不是端到端吞吐提升倍数。这个口径在对外表述时必须保留。

### 5.4 KDA 融合算子链

KDA 是“状态型”算子链，不能只补一个 attention kernel。项目最终接入的路径包括：

- 因果短卷积 prefill/update；
- chunk prefill 与 recurrent decode；
- RMSNorm + sigmoid gate；
- state gather/scatter；
- 状态页选择、PAD 槽归一和跨块状态迁移。

适配中重点处理了四类契约问题：

1. 算子要求 4-D 零 permute ABI，不能把上游的 3-D/转置布局直接传入。
2. state 布局在插件与 kernel 之间互为转置，必须显式统一。
3. 芯片要求每条逻辑行独立 512 B 物理行，view 操作数需要物化；后来通过 ABI 扩展逐步消掉。
4. `PAD_SLOT_ID` 必须归一为 `-1`，否则会触发断言甚至让设备不可用。

在算子验收阶段还定位到几类真实缺陷：长序列触发设备卡死、wrapper 整池搬运、kernel 不读 stride、aten 侧 scalar 打包错误导致输出恒为 0（即使用户态看起来“调用成功”）。修复后，34 层 KDA 在真机上零 fallback，recurrent decode 约 0.49–0.60 ms/call，state gather+scatter 约 0.76 ms/步（页表实现约 5.6 ms/步），并具备 graph 捕获条件。

### 5.5 KPool 与 NoPE-512 稀疏 MLA

KPool 的数据流可以简化为：

```text
4 个 token 的 K/gate
  → 逐通道 softmax 加权压缩成一个 pooled key
  → 每个 query head 与 pooled key 做点积
  → ReLU
  → 按 head 权重汇总
  → pool 级 TopK 512
  → 展开为最多 2048 个 token
  → 附加最多 3 个 tail token
  → 输出按 2176 对齐
```

这里有一个容易被现有算子“形状正确但语义错误”掩盖的点：GLM5 的分数是“逐 head 点积 → ReLU → 按 head 加权”，而现有 MQA 算子的参考实现是“先加权 query 再点积”。两者一般不等价。项目先用插件 native 路径保证语义，再接入 KPool 融合链：更新/压缩、select/top-k/expand、slot mapping 三条路径覆盖 11 个 indexer 层。

NoPE 稀疏 MLA 同样按 512 维 latent 单独建路径：它不走 576 维（512+64）旧布局，而是直接读分页的 512 维 cache。算子验收中，`Q=2048/H=8/K=2176` 的稀疏 MLA 从 native 约 2,011 ms 降到约 31 ms，`Q=1` decode 从约 126 ms 降到约 0.6 ms；这是算子组交付、本人验收接入的系统结果。

长上下文性能的历史退化已定位到 KPool select/top-k 融合算子本身：该算子随可见池数量增长，存在明显的扫描/并行效率问题，是之前长上下文变慢的主要原因。后续优化完成后，该算子将不再成为长上下文的性能瓶颈。6.3 节给出的量化预估是：4K/16K/32K prompt TTFT 约 `24.7/97.2/194.7 s`，单请求 4K/16K/32K decode 约 `104/115/133 ms/token`，32 请求 32K decode 约 `1.24 s/step`、聚合约 `26 tok/s`。

### 5.6 mHC、MoE 与 AWQ

- mHC 接入 pre、post 和 fused post+pre 三条路径；子层 RMSNorm 放进融合 kernel，避免额外的 `[N,4096]` 读写。算子组交付后，mHC pre 在 N=1 时从 2.845 ms 降到 0.407 ms（约 7×），post 在 N=4096 时从 213.6 ms 降到 4.5 ms（约 47.9×）；这些是 kernel 侧收益，本人负责验收和接入。
- AWQ MoE 修正了 `qzeros` 的 colpack 布局，避免错误复用 `qweight` 的 rowpack/col-major 路径；`swiglu_limit` 从模型配置动态读取 10.0，不再硬编码 7.0。
- 第一阶段 MoE router 使用 BF16 logits，而模型配置要求 FP32。该差异被明确记录为已接受的精度风险，不能表述为“FP32 router 已完成”。
- AWQ MoE 的主机侧布局逆变换/反量化辅助函数是生产必需代码，不是调试探针；发布清理时保留。

### 5.7 Host/device 流水线与集合通信

profile 显示，芯片固定成本主要来自“下发次数”，而不是数据量：

- 单次 H2D 固定成本约 44 µs；900 次小拷贝约 39.9 ms，合并成一次大拷贝约 0.10 ms，差约 400×。
- decode 每步曾有约 3,888 次 kernel 下发、约 389.5 ms 设备空洞、约 204 次 view 物化，以及 KPool 两处长阻塞 D2H。

对应优化包括：

- 合并 metadata H2D，常驻 pinned/device buffer，减少重复小拷贝；
- 去掉 KPool 每层 `rows.tolist()` 的阻塞读回，改为 device 侧计划和持久 buffer；
- 静态参数加载期预物化，每步元数据只 pack 一次；
- 逐项消掉 KDA decode 的 view 物化，最终该条目从 204 次/步降到 0；
- 用融合算子减少 kernel 数，而不是在 Python 层继续“优化编排”。

效果是：kernel 数从 3,888 降到 1,982（约 -49%），设备空洞从 389.5 ms 降到 29.1 ms（约 -92.5%），长阻塞 D2H 从约 223 ms/步降到没有 >0.5 ms 的阻塞项。集合通信方面，拆分后确认约 70% 的 allreduce 时间是“等其它 rank 到齐”，ring 本体每次仅约 0.55 ms；由于没有计算/通信重叠，屏障到达偏斜仍是主要残余瓶颈。

### 5.8 Graph 捕获与 padding

发布配置采用 `FULL_DECODE_ONLY`：decode 走 graph，prefill/chunked prefill 保持 eager。启动日志显示捕获 35 张 decode graph，最大 capture size 为 256。这里没有使用 piecewise graph，因为当前芯片构建把 `torch.compile` 做成透传，piecewise 没有可用子图。

Graph 首次 8 并发验证暴露了一个典型边界问题：

- 实际只调度 6 个 decode token，但 graph bucket 为 8；
- `q` 被补齐到 8 行，尾部 state 槽为 `NULL`；
- 旧 metadata 把 padding 的 `query_start_loc` 都填成 6，模型侧又按 `num_decodes + 1` 截断，kernel 看到最后一个边界 6 与 padded N=8 不一致，报 `query_start_loc must cover [0, N]`。

修复分两处：在 attention metadata 构造时把尾部 PAD token 分配给 `NULL` state 槽，形成 `[0,1,2,3,4,5,6,7,8]` 的完整边界；KDA 侧不再截断 `cu_seqlens`，直接传完整 persistent buffer，符合 4-D ABI。修复后 1/2/8/16/24/32 并发 graph 全部成功，graph 与 eager 同输入逐字一致。

性能上，单请求 decode gap 从约 `110.27 ms` 降到 `67.64 ms`；8 并发吞吐约 `+9%`，32 并发基本持平；64 并发因实际 38 个请求落入 40 请求槽位的 bucket，产生约 5% padding 成本，graph 反而略慢。开启 replay cache 后，单并发三档 `gap_p50` 再降约 9%–10%，TTFT 变化小于 1%。

### 5.9 NaN 故障：两个独立根因

故障现象是服务正常启动、`/health` 返回 200、请求也返回 HTTP 200，但模型输出固定为 `!!!!!!!!` 一类无意义 token，hidden states/logits 大量为 NaN。

**阶段 A：KDA state pool 地址溢出**

- 芯片编译目标下指针和 `unsigned long` 为 32 位；KDA 把以 128 B 为单位的地址乘以 32 转成 4 B word 地址，超过 16 GiB 后发生回绕。
- 模型 state 约在 34 GiB，gather 实际没有写输出，layer 0–2 读到 allocator 残留 NaN；scatter 复用同一路径，也存在同样风险。
- 低地址单测全部通过，不能排除该问题；直到用 sentinel、强制 mask、地址阈值实验和 18 GiB 占位测试，才把边界锁定在 16 GiB。
- 修复是把 metadata block、word offset、地址转换和 gather/scatter 共用 copy 路径统一改为 64 位，并补高 HBM 地址回归。

**阶段 B：sparse MLA null block 被 warmup 污染**

- 修复 A 后，首个 NaN 从 layer 0 移到 layer 3，说明 KDA 已恢复，但第二个根因暴露。
- V1 dummy sampler warmup 以全 0 `slot_mapping` 执行 11 个 sparse MLA 层写入；dummy latent 全为 NaN，把 null block 0 写脏。
- 真实请求只分配 block 1 及之后，但 sparse kernel 会把无效 top-k 索引映射到 block 0，并在 mask 之前执行 QK GEMM；405,248 行 cache 中只有 slot 0 是 NaN，也足以污染整批输出。
- 修复是在所有 dummy/warmup 结束后、服务 ready 前再次清零 null block 0；后续加固建议包括 dummy run 使用 `PAD_SLOT_ID(-1)` 或 scratch block，以及在读取端对无效索引先清零。

最终服务 8 个 worker 正常 ready，短请求和 32-token 请求返回正常文本。这个案例沉淀成了通用 SOP：HTTP 200 不等于模型正确；先找第一个 NaN 的生产者，再按“地址宽度、cache 生命周期、无效索引、warmup”逐层排除。

## 6. 结果

### 6.1 功能与质量

| 验证项 | 结果 |
|---|---|
| 健康检查、模型列表、单请求、流式输出 | HTTP 200；单请求 20 prompt / 32 output 约 2.107 s |
| 8 并发 | 3.751 s 完成 256 个输出 token，约 68.24 output tok/s |
| 长上下文功能 | 3,610 prompt / 8 output 功能通过；16K/32K chunked prefill 在发布前矩阵中通过；历史时延数字不作为 KPool 优化后的最终长上下文表现 |
| Graph 并发矩阵 | 1/2/8/16/24/32 并发全部成功；32 并发及以下无 worker died/断言失败 |
| Graph 与 eager | 相同 prompt、`temperature=0`、32 tokens 输出逐字一致 |
| 权重映射 | 111,346 个 key 完成静态审计；无 missing/unexpected/重复目标 |
| 定向回归 | `161 passed, 10 subtests passed`；`compileall`、`ruff check/format`、`git diff --check` 通过 |
| NaN 回归 | 高地址 gather/scatter 单测通过；服务级短/长请求输出正常 |
| 端到端精度 | AIME26 30/30 全对，与官方表现指标一致（云服务团队生产验证） |

> 说明：16K/32K 早期矩阵是在 KPool select/top-k 走 native 回退时采集的；最终发布路径已统一走融合链。KPool select/top-k 的长上下文耗时问题已定位，优化后的长上下文预估值见 6.3 节。

### 6.2 性能结果

**首次可用与后续优化（不同口径分开看）**

| 阶段 / 场景 | 结果 | 口径 |
|---|---|---|
| 状态拆池前 | prefill 约 179 s/step；decode 约 120 s/step | 单请求、同一短 prompt、插件内插桩 host 单步 |
| 状态拆池后 | prefill 约 5.7 s/step；decode 约 2.6 s/step | 同上；不是端到端吞吐 |
| 10-10 干净服务基线 | 约 233 ms/token（4.4 tok/s） | 1,223-token prompt、无 profiler、eager |
| 10-22 干净 decode | 约 96.9 ms/token（10.32 tok/s）；p50 96.35 ms、p90 102.91 ms | 单请求 64 output、无 profiler、无 kernel log |
| 发布 smoke（历史） | 单请求 2.107 s；8 并发 68.24 tok/s；3,610 prompt 72.419 s | 生产配置、graph 默认开启；长上下文历史时延不用于 KPool 优化后的最终评估 |

**Graph 与 replay cache**

| 场景 | 并发 | Graph gap p50 | Eager gap p50 | Graph 吞吐 | Eager 吞吐 |
|---|---:|---:|---:|---:|---:|
| 49 prompt + 96 output | 1 | 67.64 ms | 110.27 ms | 14.11 tok/s | 8.83 tok/s |
| 49 prompt + 96 output | 8 | 103.49 ms | 112.86 ms | 65.38 tok/s | 59.95 tok/s |
| 49 prompt + 64 output | 32 | 222.04 ms | 222.88 ms | 100.08 tok/s | 100.55 tok/s |
| 49 prompt + 64 output | 64 | 277.89 ms | 263.59 ms | 92.10 tok/s | 99.12 tok/s |

开启 graph replay cache 后，三档单并发 `gap_p50` 从 `63.78/74.42/80.13 ms` 降到 `57.58/67.61/72.31 ms`，约 -9%，对应吞吐约 16.53/12.13/2.67 tok/s；TTFT 变化小于 1%。注意这些数字的 prompt/输出长度不同，不能把 96.9 ms、57.6 ms 和 15–17.5 tok/s 混成同一个 SLA。

**流水线指标（profiled decode step）**

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| decode 每步 view 物化 | 204 次 | 0 | 清零 |
| kernel 下发次数 | 3,888 | 1,982 | -49.0% |
| 设备空洞 | 389.5 ms | 29.1 ms | -92.5% |
| 屏障到达偏斜累计 | 220.5 ms | 90.4 ms | -59.0%（目标 ≤20 ms，未达成） |
| 长阻塞 D2H | 约 223 ms/步 | 无 >0.5 ms 阻塞项 | 长阻塞消除 |
| profiled decode 步长 | 602.6 ms | 140.6 ms | 约 -77% |

### 6.3 上下文、容量与长上下文预估

- 模型声明上限为 1,048,576 tokens，但 8 卡实际可用长度由 KV block 预算决定。
- 0.80 显存利用率下，启动日志为 1,411 blocks，理论上限约 **171,776（约 171K）**；达到该长度时并发能力为 1.00x，没有余量。
- 0.95 显存利用率约 1,883–1,891 blocks，理论约 **232K**；仅有线性外推，未作为生产配置验证。
- 当前常规验证配置是 4096；131,072 曾在 0.85 显存利用率、eager 路径下完成验证。
- 容量上限不等于可接受时延：prefill 仍随 token 数增长，长上下文端到端规划需要同时核对 KV block 预算和实际输入长度。

**KPool select/top-k 优化后预估值**

以下数字是根据现有 fast/native 微基准和模型级时序拆解得到的预估值，不是新增实测。估算采用保守理想假设：优化后的 select/top-k 至少达到 native 批处理路径的吞吐，不再随上下文增长而恶化；不额外假设超过 native 的融合收益。

如果后续实现进一步采用 query tile × pool tile 并行、双计算核复用和 top-k/expand 融合，实际值应低于表中预估；表中取 native 路径作为保守下限。

2026-11-02 已落地第一轮 kernel 常数优化（head 8 路 unroll、`scale × head_weight_scale` 预折叠、block table/assert 外提）：4K/16K/32K 等 5 个长上下文场景实测约快 7.1%–7.5%，输出 hash 不变；详细见《KPool select/top-k 根因分析与优化方案》。

| KPool select/top-k 场景 | 首轮优化前 fast | 优化后预估 | 改善 |
|---|---:|---:|---:|
| 4K prefill（11 层） | 85.2 s | 4.8 s | 约 17.8× |
| 单请求 4K decode | 57.5 ms/step | 7.5 ms/step | 约 7.7× |
| 单请求 16K decode | 227.6 ms/step | 18.3 ms/step | 约 12.4× |
| 32 请求 32K decode | 7.15 s/step | 1.02 s/step | 约 7.0× |

**整模长上下文预估值（保守下限）**

单请求 prefill TTFT：

| prompt 长度 | 预估 TTFT | 估算依据 |
|---:|---:|---|
| 4K | 24.7 s | native 路径 4K 整模实测 |
| 8K | 约 49 s | 按 4K 线性外推 |
| 16K | 97.2 s | native 路径 16K 实测 |
| 32K | 194.7 s | native 路径 32K 实测 |
| 131K | 约 797 s（13.3 min） | 按 16K/32K 线性外推 |

decode 预估：

| 场景 | 预估 step / token 间隔 | 预估吞吐 | 估算依据 |
|---|---:|---:|---|
| 单请求 4K context | 104 ms/token | 9.6 tok/s | 96.9 ms 基底 + 7.5 ms selection |
| 单请求 16K context | 115 ms/token | 8.7 tok/s | 96.9 ms 基底 + 18.3 ms selection |
| 单请求 32K context | 133 ms/token | 7.5 tok/s | 96.9 ms 基底 + 36.5 ms selection（线性外推） |
| 32 请求 8K context | 581 ms/step | 55 tok/s（聚合） | 222 ms 基底 + 359 ms selection |
| 32 请求 16K context | 804 ms/step | 40 tok/s（聚合） | 222 ms 基底 + 582 ms selection |
| 32 请求 32K context | 1.24 s/step | 26 tok/s（聚合） | 222 ms 基底 + 1,015 ms selection |

预估中的 decode 基底取干净短上下文单请求 96.9 ms/token，以及 32 并发 graph 的 gap p50 222 ms/step；假设 MoE、KDA、稀疏 MLA 等非 selection 部分在长上下文下保持同量级。以上为量级预估，不作为 SLA 或新增测试结论。

### 6.4 工程资产

- 形成 50+ 份设计、验证、排障和交付文档，40+ 个可复现探针脚本，以及完整的测量原始数据。
- 固化启动/部署指南、Graph 验证记录、发布前功能/性能矩阵、NaN 排查 SOP、KPool 性能量化、算子需求与验收材料。
- 发布清理后，生产分支只保留模型、缓存、算子接入、Graph 和契约测试；探针、历史验证和调试代码归档到工作文档目录。

## 7. 已知边界与后续工作

| 边界 | 影响 | 后续方向 |
|---|---|---|
| MoE router 使用 BF16 logits | 可能改变临界 expert 的 Top-8 选择 | 扩展 fused MoE 的 FP32 routing 接口并做整层对照 |
| 仅 V1 runner | V2/Triton 路径未验证 | 具备 Triton 环境后单独验证 graph、metadata 和 KPool |
| Graph 只覆盖 uniform decode | prefill/chunked prefill 仍是 eager；非 2 次幂并发会落入更大 bucket | 评估业务热点 bucket、prefill graph 和更细的捕获策略 |
| 64 并发 graph 约 -5% | 38 个真实请求落入 40 请求槽位的 bucket，padding 成本 | 增加 38/39 等 bucket，或按业务并发分布重新设计捕获点 |
| KPool select/top-k 历史长上下文退化 | 已定位为该融合算子随可见池数量增长导致的主要瓶颈；首轮常数优化已带来约 7.5% 收益，但数量级提升仍需矩阵化/tile 重写 | 优化后的长上下文预估值见 6.3 节；容量仍按 block 预算核算 |
| 集合通信无计算重叠 | 屏障 root-critical 等待仍是 decode 主要残余瓶颈；90.4 ms 尚未达到 ≤20 ms 目标 | 多流重叠、通信/计算流水、rank 间 host 对齐和 runtime 固定成本优化 |
| 长上下文容量是配置上限而非 SLA | 171K 是 0.80 下的理论容量，232K 只是外推；生产可用长度仍由启动时 KV block 和实际 `max-model-len` 决定 | 按生产配置核对 KV block、TTFT 和并发预算 |
| MTP、多模态、FP8 未纳入发布 | checkpoint 中存在相关结构，但当前生产启动使用 `--language-model-only` | 作为后续独立阶段，不与本次文本 BF16 发布混写 |

## 8. 复盘

### 8.1 做对的地方

1. **先建模型账本，再写代码**：把 KDA、KPool、MLA、mHC、MoE 的输入输出、状态和缓存边界先画清楚，后续排障才能快速判断“是模型语义、算子契约还是布局问题”。
2. **用分层隔离代替猜测**：从端到端 step → 模块 → 算子 → kernel → ABI 逐层收敛；拆池问题就是通过“连续池 vs padded view”的隔离微基准，把“kernel 慢”改判为“布局税”。
3. **重视边界值和物理约束**：16 GiB 地址回绕、非 512 B 对齐、`B==128`、graph padding、null block 0，都是常规小数据测试覆盖不到的边界；补上高地址、非连续、padding 和生命周期测试后，问题才真正闭环。
4. **用 sentinel 和阶段切分验证根因**：先确认 gather 是否写输出，再看首个 NaN 的层号是否后移；这比“反复跑整模型看输出”更快，也避免了把两个独立根因混成一个。
5. **性能优化围绕关键路径，而不是局部 microbench**：最终真正有效的是拆池、减少下发次数、去掉阻塞 D2H、融合算子和 graph；单个 kernel 的局部优化如果没有进入真实路径，收益会被 host 开销吃掉。
6. **测试要证明“真的走了 kernel”**：native/fused 两边都回退时对比也能“通过”，必须用 monkeypatch 计数、命中日志或探针断言 fast path 实际生效。
7. **发布前做减法**：保留必要契约测试和 CPU 辅助函数，删除 native/fallback、调试分发和探针，降低生产路径分支数。

### 8.2 可以做得更好的地方

- **线上验证与研发验证需要持续衔接**：AIME26 30/30 已完成端到端精度验证；后续扩展数据集或模型版本时，仍应固定参考口径和回归集。
- **长上下文性能应更早拆分到单算子**：KPool select/top-k 的历史退化已定位为该算子的扫描/并行效率问题；后续以 6.3 节的预估值作为门槛，不再用历史微基准作为门禁。
- **边界测试应前置到算子验收**：高地址、非连续、512 B 对齐、`B==128`、PAD/null slot 都应在算子交付时成为必测项，而不是等整模型 NaN 或 graph 失败后反推。
- **发布清理可以更早开始**：fallback 和 monkey-patch 在 bring-up 阶段很有价值，但长期并存会增加路径分叉和测试假绿风险。
- **性能口径要统一管理**：profiler/无 profiler、单步/端到端、单请求/并发、prompt 长度不同都会导致数字不可比；后续应把测量条件写进每次结果表。

### 8.3 可复用经验

- 异构有状态模型接入：先建立“计算图 + 状态生命周期 + 缓存布局”三张图，再拆算子。
- 非 Triton 平台的性能排障：优先怀疑布局、对齐、host 往返和下发次数，而不是先怀疑算力。
- 多根因故障处理：每修一个根因就重新建立“首个异常层/首个异常算子”基线，用中间结果证明修复有效。
- Graph 适配：padding、动态 shape、持久 buffer 和 metadata 边界要一起验证；graph 不是“打开开关”而是新的执行契约。
- 算子协作：需求要写清输入/输出/布局/空槽语义/精度/性能判据，验收要证明真实路径命中，而不是只看 shape。

## 9. 个人贡献与能力体现

如果用于述职、绩效或面试，可以突出以下五点：

1. **端到端 Owner 意识**：从模型注册、权重映射、缓存布局、算子接入到 Graph、NaN 和发布验证，独立把插件侧链路打通；模型最终由云服务团队正式生产上线，AIME26 30/30 与官方表现一致。
2. **跨栈定位能力**：能同时读模型 Python、PyTorch ATen、算子 ABI、kernel 头文件和运行时日志，把“模型输出不对”拆成可验证的地址、布局、生命周期和索引问题。
3. **性能工程能力**：不是堆优化点，而是用 profile/微基准建立因果链；拆池、metadata 合并、D2H 消除和 graph 都直接对应关键路径。
4. **复杂系统取舍能力**：在发布窗口内明确哪些必须做、哪些可以接受、哪些留到后续；例如把 KPool 长上下文退化收敛到 select/top-k 单算子，并在 6.3 节给出优化后的量化预估，而不是把历史退化当成最终结论。
5. **协作与交付能力**：把算子需求、复现脚本、验收结论和 SOP 沉淀成团队可复用资产；与算子、PyTorch 后端、运行时和通信团队形成闭环。

## 10. 两分钟口述版

我们要在 8 卡自研 AI 加速芯片上跑 GLM-5.3-Flash。这个模型不是普通 Transformer：它有 34 层 KDA 线性注意力、11 层 KPool 稀疏 MLA、四路 mHC 残差和 42 层 AWQ MoE，上游实现还依赖 Triton 和 TileLang，而我们的芯片没有这些。

我在项目里独立负责 vLLM 插件侧的适配：先把模型、权重和 KV/state cache 接进来，再用 native 路径绕开 Triton；同时把融合算子需求拆给算子组，交付后逐项验收、接入并保留回退。最难的问题不是公式，而是状态布局：KDA 的 conv 和 recurrent state 被放在同一个字节页里，切出来是非连续 view，芯片后端会退化成 host 逐元素搬运。一个 15-token prefill 单步要 179 秒，decode 要 120 秒。我用 py-spy、隔离微基准和 profile 定位到这是“布局税”，把状态拆成两个连续池后，单步降到 5.7 秒和 2.6 秒。

之后我继续按 profile 做优化：接入 KDA、KPool、mHC、512-NoPE 稀疏 MLA 的融合路径，合并 metadata H2D、去掉阻塞 D2H、减少 kernel 下发，并把 decode 做成 `FULL_DECODE_ONLY` graph。最终干净 decode 约 96.9 ms/token，graph 单请求 decode gap 从约 110 ms 降到约 68 ms，开启 replay cache 再降约 9%；发布 smoke 单请求 2.1 秒、8 并发约 68 tok/s，插件定向回归 161 项通过。项目最终由云服务团队正式生产上线，AIME26 端到端 30/30，与官方表现指标一致。KPool 长上下文的历史退化已定位到 select/top-k 融合算子；优化后预估 4K/16K/32K prompt TTFT 约 24.7/97.2/194.7 秒，32 请求 32K decode 约 1.24 秒/step、聚合约 26 tok/s。最后还定位了全 `!`/NaN 的两个独立根因：KDA 超过 16 GiB 的 32 位地址回绕，以及 warmup 污染 sparse MLA null block 0，两个都修复并沉淀成 SOP。

这个项目让我最满意的是：不仅把模型“跑起来”，还把性能、正确性、边界和发布路径都做成了可解释、可复现、可交接的工程结果。

## 11. 证据索引

| 文档 | 用途 |
|---|---|
| 《模型服务使用指南》 | 生产启动参数、环境变量、Graph 配置、上下文上限和常见问题 |
| 云服务团队生产反馈 | 正式生产上线、AIME26 30/30、与官方表现指标一致 |
| 《代码清理与合并说明》 | 发布分支改动、删除的 native/fallback、测试结果和合入注意事项 |
| 《Graph 捕获验证》 | Graph 修复、并发矩阵、graph/eager 对比和 replay cache 复测 |
| 《NaN 问题排查总结与 SOP》 | 两个 NaN 根因、证据链、修复和通用排查流程 |
| 《发布前功能与性能验证》 | 功能矩阵、快路径命中和流水线量化指标 |
| 《KPool select/top-k 性能量化》 | 历史性能量化、长上下文瓶颈归因和后续优化方向 |
| 《适配研究》 | 模型结构、KDA/KPool/mHC 公式、缓存和算子缺口 |
| 《联调问题与修复记录》 | 首次真机启动、两次 OOM、AWQ 布局、状态迁移等联调问题 |
| 《阻断点 2–7》 | KDA、MLA、mHC、AWQ、权重映射和 KPool 的静态缺口 |
| 《拆池方案》 | Mamba/KDA 状态池拆分、页表实现和回退开关 |
| 《性能优化点清单》 | P0–P2 优化项、归因数据、覆盖关系和执行顺序 |
| 《算子覆盖核查与接入清单》 | 18 项候选需求的分类、已有算子覆盖、验收与接入状态 |
| 《集合通信同步归因》 | allreduce 等同步与通信本体拆分、host 侧空洞和屏障偏斜 |

## 12. 对外表述边界

| 建议表述 | 不建议表述 |
|---|---|
| 已由云服务团队正式生产上线；AIME26 30/30，与官方表现指标一致 | 项目组自行承担了线上运营和长期 SLA |
| 独立负责插件侧适配、性能优化、NaN 排障和发布收敛 | 所有 kernel 均由本人实现或优化 |
| 状态拆池使单步 host 耗时从约 179 s/120 s 降到 5.7 s/2.6 s | 端到端吞吐提升 31×/46× |
| Graph 单请求 decode gap 约 110→68 ms；replay cache 再降约 9%–10% | 所有并发和所有输入长度都获得同等收益 |
| 0.80 显存下理论上下文上限约 171K，131K 曾在 eager/0.85 下验证 | 1M 上下文已可直接生产服务 |
| 定向回归 161 项通过，权重映射 111,346 key 审计通过；AIME26 30/30 | 用单元测试通过代替端到端精度结论 |
| KPool select/top-k 长上下文瓶颈已定位，优化后预估值见 6.3 | 当前历史退化数字就是优化后的最终长上下文表现 |
| KDA/KPool/mHC kernel 由算子组实现，本人负责需求、验收和接入 | 把 kernel 性能优化归为个人成果 |
| 当前发布为文本、BF16、V1、decode graph 路径 | 已支持 MTP、多模态、FP8、V2 和计算/通信重叠 |
