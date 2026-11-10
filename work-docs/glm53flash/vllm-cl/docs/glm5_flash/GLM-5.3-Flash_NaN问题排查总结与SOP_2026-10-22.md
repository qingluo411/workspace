# GLM-5.3-Flash NaN 问题排查总结与 SOP

> 日期：2026-10-22
>
> 范围：GLM-5.3-Flash + vLLM V1 + vLLM-CL + DLC KDA + DLC sparse MLA
>
> 结论：本次不是一个单点故障，而是**两个相互独立的缺陷在修复过程中先后暴露**：
>
> 1. **KDA state pool 32 位地址运算溢出**：模型状态池位于 HBM 16 GiB
>    以上时，gather/scatter 的地址换算被截断，导致 layer 0–2 的 KDA
>    `initial_state` 读到 allocator 残留 NaN。
> 2. **sparse MLA null block 0 被 warmup 污染**：修掉前一个问题后，首个 NaN
>    移到 layer 3；V1 dummy sampler warmup 以全 0 `slot_mapping` 把全 NaN latent
>    写进 block 0，而 sparse kernel 会把无效 top-k 索引映射到 block 0，并在
>    mask 前执行 QK GEMM，最终污染 layer 3–43。
>
> 两个根因必须分别修复。只修 KDA 时输出仍为全 `!`；只清 sparse null block
> 则会漏掉低层 KDA NaN。

## 1. 结论摘要

- **现象**：服务能正常启动、`/health` 返回 200、请求也返回 HTTP 200，但模型输出
  固定为 `!!!!!!!!` 一类无意义 token；hidden states / logits 大量为 NaN，Inf 为 0。
- **阶段 A 根因**：DLC 编译目标下 `unsigned long` 和指针均为 32 位。KDA state
  pool 把以 128 B 为单位的 `DLCTensor.address` 乘以 32 转成 4 B word 地址时，
  在 HBM 偏移超过 16 GiB 后发生 32 位回绕。模型 state 约在 34 GiB，因此 gather
  实际未写输出；scatter 复用同一地址路径，存在同样风险并已一并修复。layer 0–2
  的 NaN 是 allocator 旧内容，不是算术产生的。
- **阶段 A 修复**：`DLC_Custom_Kernel/dlc_kernels/attention/kda/kda_state_pool.h`
  中的 metadata block、word offset、地址转换、gather/scatter 共用 copy 路径全部
  改为显式 64 位。
- **阶段 B 根因**：V1 dummy sampler warmup 以全 0 的 `slot_mapping` 执行了 11 个
  sparse MLA 层的 cache 写入；dummy latent 全为 NaN，因此 null block 0 从全零
  变成全 NaN。真实请求只分配 block 1 及之后，但 sparse kernel 会把
  `topk_indices < 0` 或越界索引映射为 0，并在 mask 前读取 block 0 参与 QK。
- **阶段 B 修复**：在 `DLCWorker.compile_or_warm_up_model()` 的所有 dummy warmup
  完成后、对外服务前，再次清零 null block 0。
- **验证**：阶段 A 修复后，高地址 gather/scatter 回归通过，模型 layer 0–2 全部
  有限，首个 NaN 从 layer 0 前移到 layer 3；阶段 B 修复后 8 个 TP worker 正常
  ready，短请求和 `max_tokens=32` 请求均返回正常文本，不再输出整段 `!!!!!!!!`。

## 2. 运行环境与复现方式

| 项目 | 值 |
|---|---|
| 硬件 | 8 × TYD 真机 |
| 模型 | `/data/models/GLM-5.3-Flash-Mix` |
| 推理框架 | vLLM V1 model runner |
| 并行 | TP=8，PP=1 |
| 启动参数 | `--enforce-eager --block-size 256 --max-model-len 2048` |
| 服务端口 | `18005` |
| 关键插件 | `vllm-cl` 的 `dlc,dlc_patches` |
| 设备版本 | 驱动 API 24 / Synapse API 26；独立 DLC 测试需跳过版本检查 |

独立复现 DLC 算子时使用：

```bash
export DLC_SKIP_VERSION_CHECK=1
export DLC_TPU_VERSION=2
export DLC_TEST_DEVICE=dlc:0
```

最小复现请求：

```bash
curl -sS --max-time 600 \
  http://127.0.0.1:18005/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "GLM-5.3-Flash-Mix",
    "messages": [{"role": "user", "content": "只回复：启动成功"}],
    "max_tokens": 8,
    "temperature": 0,
    "stream": false
  }'
```

故障态返回示例：

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "!!!!!!!!"
      },
      "finish_reason": "length"
    }
  ]
}
```

## 3. 问题现象与特点

### 3.1 现象

1. 服务启动、模型加载、KV cache 分配和 warmup 都没有报错。
2. `/health` 正常返回 200，8 个 TP worker 均在。
3. 真实请求同样是 HTTP 200，但输出稳定退化为全 `!` 或类似重复标点。
4. 中间激活和 logits 大量为 NaN，而 Inf 计数为 0。
5. PCIe/初始化卡死通过 `cltechpd_clnt -s --full` 复位后，服务能启动，但 NaN
   继续存在。因此硬件复位与 NaN 是两个独立问题。
6. 初始首个 NaN 出现在 layer 0 的 KDA `initial_state`；修复 KDA 地址问题后，
   layer 0–2 恢复有限，首个 NaN 才移动到 layer 3 的 sparse MLA 输出。
7. 模型配置中 layer 0–2 是 KDA linear attention，sparse MLA 层为
   `[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]`。这个结构边界是拆分两个根因的
   关键：层号变化证明了修复有效，而不是“第一个修复没有生效”。

### 3.2 问题特点

| 特点 | 本问题中的表现 | 排查意义 |
|---|---|---|
| 静默失败 | 不抛异常，HTTP 仍为 200 | 不能只看接口状态，必须检查 logits/hidden states |
| 确定性 | `temperature=0` 时稳定复现 | 可以先做最小、可重复的精确复现 |
| 全链路放大 | 某一层一旦读到 NaN，残差流后续层全部 NaN | “第一次看到 NaN 的层”不等于“NaN 的生产者” |
| 地址阈值相关 | 低 HBM 地址全部正常，跨过 16 GiB 后完全不写；模型 state 约在 34 GiB | 单元测试只覆盖低地址会形成假阴性 |
| 多根因串联 | 修完 KDA 后首个 NaN 从 layer 0 移到 layer 3 | 每修一个根因都要重新建立“首个 NaN 层”基线 |
| 静默未写 | gather 输出先填 42，调用后仍是 131072 个 42 | sentinel 可区分“算错”和“根本没写” |
| 局部头/切片异常 | 首个 `initial_state` 只有 head 5 全为 `0x7fc00000` qNaN，其余 head 是旧状态位型 | 单片异常更像地址错误或残留内存，不像统一数值溢出 |
| 局部数据污染 | 405248 行 cache 中只有 slot 0 一行全 NaN | 说明不是整块 cache 未初始化，而是单点脏写 |
| 路径相关 | 合法 top-k 槽位数据正常，无效 top-k 槽位映射到 0 后才读 NaN | 需要同时检查有效数据和 padding/无效索引路径 |
| warmup 相关 | 启动后即存在，真实请求自己并未写坏 block 0 | 必须审计启动、dummy run、cache 生命周期 |
| mask 时序敏感 | NaN 在 QK 点积阶段已经传播，softmax mask 来不及补救 | “后续会 mask”不能防止 NaN 污染 |

## 4. 定位过程

整体时间线（后续小节展开证据）：

1. 服务/接口正常但输出全 `!`，用逐层探针确认是 hidden states/logits NaN；
2. 首个 NaN 收敛到 layer 0 的 KDA `dlc_kda_chunk_prefill.initial_state`；
3. 低地址 gather、mask 变体和 1 万次压力测试正常，形成第一次误判；
4. 用 sentinel A/B/C/D 发现真实调用下 gather 根本没写输出；
5. 地址扫描发现跨过 16 GiB 后确定性失效，确认 32 位 word 地址乘法溢出；
6. 修复 KDA helper 为 64 位并重建/重启，layer 0–2 恢复有限，首个 NaN 移到 layer 3；
7. 保存 sparse MLA 真实 payload，复现“slot 0 为 NaN 时整行 NaN”；
8. 审计完整 cache 与写入探针，锁定启动 dummy warmup 污染 null block 0；
9. 在所有 warmup 后清零 block 0，完成服务级短/长请求验证。

### 4.1 先确认是数值问题，而不是接口问题

最初服务健康检查、请求返回码、tokenizer 和调度都正常。真正异常的是 hidden
states、logits 和采样结果。由此把问题限定为：

> 模型前向或 KV/state 数据中出现 NaN，而不是 API、网络或 tokenizer 故障。

同时确认：

- 模型权重加载后未发现 NaN/Inf；
- 输入 embedding 和早期线性层输出可保持有限值；
- NaN 计数远大于 Inf 计数，排除单纯的 FP16/BF16 overflow 到 Inf 后再计算。

### 4.2 建立逐层 NaN 探针，但避免被 warmup 误导

在 decoder layer、attention、MLP、MoE 等模块上增加输入/输出统计，记录：

- NaN / Inf 数量；
- `absmax`；
- shape、dtype、模块名；
- 当前是 warmup、prefill 还是 decode。

这一步确认了 NaN 会通过残差连接迅速扩散到后续层。但也暴露了一个重要陷阱：

> 启动阶段有多个 dummy run，输入形状和真实请求不同。如果不区分 warmup 与真实
> 请求，很容易把“第一次观察到 NaN 的层”误认为根因。

后续所有探针都增加了请求阶段、step、shape 和调用序号标记，避免混用数据。

### 4.3 首个 NaN 出现在 layer 0 的 KDA `initial_state`

对真实请求做逐层探针后，最早的生产点不是 sparse MLA，而是 layer 0 的 KDA
chunk prefill 输入：

- `state` 形状 `(1583, 8, 128, 128)`，stride
  `(131072, 16384, 128, 1)`；
- `indices=[4]`，CPU 视角 `mask=[False]`；
- 8 个 TP rank 行为一致；
- layer 0 的 `dlc_kda_chunk_prefill.initial_state` 中只有 head 5 的 16384 个
  元素全为 `0x7fc00000`（qNaN），其他 head 是旧状态位型；
- 对 state page 做 D2H 快照时该 page 全零，看起来“mask=false 本应清零”也成立。

这里有两个相互矛盾的观察：CPU 认为 `has_initial_state=False`，state page 也
全零，但 chunk prefill 实际拿到的 `initial_state` 却有 NaN 和旧数据。说明问题
位于 gather 的写入/地址路径，而不是 page 内容本身。

### 4.4 第一次误判：低地址单元测试全部正常

最初把 gather 单独抽出来测试：使用低 HBM 地址、正常 mask、输出预填 NaN，并做
约 1 万次压力循环。结果全部通过，因此一度怀疑 metadata H2D 生命周期、mask
descriptor 或 state page 清零。这个分支非常重要：

> **低地址、同进程、独立算子测试通过，不能排除地址相关的内核缺陷。**

模型是用大块 HBM 初始化后才分配 KDA state，真实地址与临时测试完全不同；只测
低地址属于覆盖缺口。

### 4.5 sentinel 对照证明 gather 根本没有写输出

在真实模型进程内对 layer 0 第一次 gather 做同调用 A/B/C/D 对照。输出先全部填
`42`，分别使用原始 mask、克隆 mask、强制 bool false、强制 int32 false 以及
强制 bool true 重放：

```text
原始 mask      eq42=131072
克隆 mask      eq42=131072
bool false     eq42=131072
int32 false    eq42=131072
bool true      eq42=131072
returned_same_ptr=True
```

即 kernel 在每一种 mask 下都没有改动输出。进一步在真实 worker 内做小 state
自检（page 1 填 7，输出填 42）也得到 `eq42=131072, eq7=0`。与此同时 kernel
launch 日志显示：

- 设备、stream 0、`capturing=False` 正常；
- `DLC_SYN_SKIP_LAUNCH=None`；
- kernel 实际执行了约 0.9–1.5 万 cycle。

因此不是 launch 被跳过、不是 mask 语义、也不是返回了错误 tensor，而是地址计算
让 DMA 读写落到了错误位置。

### 4.6 高地址阈值实验锁定 16 GiB 地址乘法溢出

用直接测试人为占用 HBM，把 state/out 推到不同地址段，得到清晰边界：

```text
正常：state=0x1000003cbc80200  out=0x1000003cbd80200
失败：state=0x100000407800200  out=0x100000407900200
      true_eq7=0  true_eq42=131072
```

前一组 HBM 偏移低于 16 GiB，后一组跨过 `0x400000000`（16 GiB）。真实模型
worker 的 state 约为 `0x100000898200200`（约 34 GiB），必然落在失效区间。
占用 18 个 2 GiB dummy 后，gather 同样完全不写。

根因在 `DLC_Custom_Kernel/dlc_kernels/attention/kda/kda_state_pool.h`：

```c
unsigned long dst_word = (unsigned long)dst->address * 32 + ...
unsigned long src_word = ... (unsigned long)src->address * 32 + ...
```

`DLCTensor.address` 以 128 B 为单位，乘以 32 转成 4 B word 地址。DLC clang
目标下：

```text
__SIZEOF_LONG__    = 4
__SIZEOF_POINTER__ = 4
```

因此 `address * 32` 在 HBM 偏移超过 16 GiB 时超过 32 位并回绕。该头文件同时
被 gather 和 scatter 使用，两个方向都必须修。修复覆盖：

- `KdaMetadataCache.block`；
- `kda_word_offset()` 的返回值与乘法；
- `kda_load_word()` 的 word/block 与地址转换；
- `kda_page_index()` 的 word 类型；
- `kda_copy_words()` 的 `dst_word`、`src_relative`、`src_word`、sentinel 掩码；
- 所有地址转换统一使用 `unsigned long long` / `ULL`。

重建并安装 `libcustom_dlc_perf_lib.so`、重启进程后，首个真实请求的 layer 0–2
全部恢复为有限值。此时 **首个 NaN 移到 layer 3**，即第一个 sparse MLA 层。这个
层号变化是阶段 A 修复有效的关键验证，也同时开启了第二个独立问题的定位。

核心教训是：**地址宽度要以目标 ABI 为准，不能用 host Linux 的 `long=64` 推断
DLC kernel；任何“地址单位转换 × 常量”的表达式都要做高位消去测试。**

### 4.7 修复阶段 A 后重新将问题收敛到 sparse MLA 边界

对 `torch.dlc_sparse_attention_kv_cache` 增加 ATen 层输入/输出探针，直接从真实
运行中保存 payload。样本参数为：

```text
q:              (5, 8, 512)
kv_cache:       (405248, 512)
topk_indices:   (5, 2176)
valid_counts:   [1, 2, 3, 4, 5]
scale:          0.0625
```

第一次调用时：

- `q` 的 NaN 数为 0，`absmax=5.875`；
- sparse op 输出 `(5, 8, 512)` 的 20480 个元素全部为 NaN；
- 后续调用连 `q` 本身也已是 NaN，说明污染已经沿残差流向上游扩散。

这说明 sparse op 不是简单“随机算错”，而是在某一条特定数据路径上系统性读到 NaN。

### 4.8 构造可复现的精确单元测例

使用真实 ATen payload 构建 `/tmp/glm5_sparse_exact_repro.py`，将 q、cache、
topk 原样搬到 DLC 设备重放，并支持对指定 slot 注入 NaN/Inf 或清零。

关键结果：

- 在干净 cache 上重放：输出 `nan=0, inf=0, absmax=1.2265625`；
- 对 payload 中的 slot 0 执行 `--zero-slot 0`：输出恢复正常；
- 对 slot 0 注入 NaN：可复现与线上一致的 NaN 输出。

由此得到强相关证据：**block 0 的内容决定 sparse op 是否输出 NaN**。

### 4.9 审计完整 cache，发现只有 slot 0 被污染

对运行中的完整 sparse cache 做 dump，共约 405248 行 / slot：

- 只有 slot 0 这一行 512 个元素全部为 NaN；
- 所有真实 top-k 选中的合法 slot 都是有限值；
- sparse op 输出仍为整块 NaN。

这与 kernel 的索引处理一致。参见
`DLC_Custom_Kernel/dlc_kernels/attention/sparse_attention/sparse_attention_kv_cache/sparse_attention_kv_cache_bf16.cpp:2040-2042`：

```cpp
bool8_128 valid = v_s32_cmp(GTEQ, idx, 0) & v_s32_cmp(LS, idx, src_shape[curdim]);
idx = v_s32_sel(valid, 0, idx);
```

无效 top-k 索引会被替换为 0，然后仍然进入后续 gather / QK 计算。也就是说，mask
只会在 softmax 阶段屏蔽这些位置，但 NaN 在此之前已经通过点积污染了整行结果。

### 4.10 追踪 cache 写入侧，锁定污染发生在启动 warmup

继续在 `torch.dlc_vllm_concat_and_cache_mla_` 外挂写探针，记录：

- cache block 0 写前/写后统计；
- `slot_mapping` 的取值；
- 写入的 latent 是否含 NaN；
- 调用发生时属于哪个阶段。

单次 dummy 写入样本：

```text
cache shape:       (1585, 256, 512)
latent shape:      (256, 512)
slot_mapping:      (256,) 全 0
latent NaN count:  131072 / 131072 全 NaN
block 0 before:    全 0
block 0 after:     512 个元素全 NaN
```

启动过程中连续观察到 11 次同类写入，对应模型的 11 个 sparse MLA 层。也就是说：

1. cache 初始分配使用 `torch.zeros`，最初是干净的；
2. `KVBlockZeroer.warmup()` 会先清零 block 0；
3. 但之后 V1 的 dummy sampler warmup 再次执行完整前向；
4. dummy token 的 `slot_mapping` 全部指向 block 0；
5. dummy latent 全为 NaN，于是 null block 0 被永久写脏；
6. 真实请求到来时，调度器只分配 block 1 及之后的 block，并正常清零这些 block；
7. block 0 作为 null block 不再有请求写入，也没有被重新清理；
8. sparse kernel 读取无效 top-k 时命中 block 0，触发最终 NaN。

### 4.11 排除项与两阶段判读

| 假设 | 结论 | 证据 |
|---|---|---|
| KDA mask/metadata 错误 | 排除 | 原始/clone/bool/int32/true 五路 mask 下 gather 输出都保持 sentinel，说明根本没写 |
| KDA state page 本身有 NaN | 排除 | page D2H 全零；NaN 出现在未初始化的 gather 输出 buffer |
| 低地址 gather 正常即可认为 kernel 正常 | 排除 | 跨过 16 GiB 后确定性失效；模型 state 约 34 GiB |
| 修完 KDA 后输出仍全 `!`，说明 KDA 没修好 | 排除 | layer 0–2 已有限，首个 NaN 移到 layer 3；这是第二个独立根因 |
| cache 分配没有初始化 | 排除 | 分配路径使用 `torch.zeros` |
| KVBlockZeroer 漏掉 sparse MLA 类型 | 排除 | block 0 在 dummy run 前确实是全零 |
| 权重中有 NaN | 排除 | 权重统计无 NaN/Inf |
| sparse attention 算术本身错误 | 排除 | 干净 cache 上输出有限且可复现 |
| 无效 top-k 会被 mask，所以不影响结果 | 排除 | mask 发生在 QK GEMM 之后，NaN 已传播 |
| PCIe 卡死导致 NaN | 排除 | 硬件复位后初始化恢复，但 NaN 仍稳定复现 |

判断两阶段时使用一条硬规则：**每修一个根因，都重新记录“首个 NaN 出现在哪一层”
以及该层的输入/输出统计。首个 NaN 向后移动，表示前一阶段确实修复，而不是把
第二阶段误判成第一阶段仍未修好。**

### 4.12 探针本身的几个坑

本次临时探针也制造过假象，排查时必须避免把探针缺陷当成模型缺陷：

- probe 函数参数名与状态字典同名，覆盖了状态对象，触发 `IndexError`；
- layer 3 的实际模块路径是
  `DLCGlm5NextMLAAttention.mla_attn.mla_attn.impl`，第一次少取一层，探针没有
  命中目标；
- 3D paged cache 被探针误按 2D 展开，统计了错误切片；
- 不带探针时曾有一次 clean run，不能据此排除地址/时序相关缺陷；要用确定性
  阈值实验和多次重启对照；
- 额外统计/转换可能改变 allocator 行为或触发无关 `SIGFPE`，探针应尽量零侵入，
  并优先保存原始 payload 后在独立进程重放。

## 5. 根因链

本次有两条独立的因果链。阶段 A 先决定 layer 0 的输入是否有限；修掉它以后，
阶段 B 才成为可见的首个 NaN 源。

### 5.1 阶段 A：KDA state pool 地址溢出

```text
模型加载并占用大块 HBM
        │
        ▼
KDA state pool 分配到约 34 GiB 的高地址
        │
        ▼
DLCTensor.address（128 B 单位）× 32 → 4 B word 地址
        │
        ▼
目标 ABI 的 unsigned long 仅 32 位
        │
        ▼
地址跨过 16 GiB 后高位回绕
        │
        ├─ gather 未写 output → initial_state 保留 allocator 旧内容/qNaN
        └─ scatter 也可能写到错误 HBM
        │
        ▼
layer 0–2 KDA chunk prefill 产生 NaN
        │
        ▼
残差流污染后续层，输出全 '!'
```

这里 NaN 不是 KDA 数学产生的，而是 **地址错误后读到了未初始化内存**。修复地址
宽度后，该链消失，layer 0–2 恢复有限。

### 5.2 阶段 B：sparse MLA null block 生命周期被破坏

```text
KV cache 分配（全零）
        │
        ▼
KVBlockZeroer.warmup() 清零 block 0
        │
        ▼
V1 dummy sampler warmup
  - slot_mapping 全 0
  - dummy latent 全 NaN
        │
        ▼
sparse MLA null block 0 被写入 NaN
        │
        ▼
真实请求分配 block 1..N，并正常清零
        │
        ▼
sparse top-k 中大量无效索引被 kernel 映射为 0
        │
        ▼
QK GEMM 在 mask 前读取 block 0 NaN
        │
        ▼
dot product 为 NaN → 整行输出 NaN
        │
        ▼
残差连接将 NaN 扩散到后续所有层
        │
        ▼
logits 全 NaN → 采样固定输出 '!'
```

阶段 B 的本质是 **有状态缓存的生命周期不变量被破坏**：block 0 是 null block，
应在整个服务生命周期内保持全零；但启动阶段的 dummy 前向把它当成了可写 block。

两条链的共同点是都与“有状态缓冲区的读写边界”有关；不同点是 A 是地址计算/ABI
问题，B 是 cache 生命周期与无效索引语义问题。因此不能用一个 `nan_to_num` 同时
掩盖，也不应在只验证一条链后宣布问题结束。

## 6. 修复方案

### 6.1 阶段 A：KDA 地址运算改为 64 位

修改 `DLC_Custom_Kernel/dlc_kernels/attention/kda/kda_state_pool.h`：把
metadata block、word offset、地址转换以及 gather/scatter 共用的 page copy
路径统一改为 `unsigned long long`：

- `KdaMetadataCache.block`；
- `kda_word_offset()` 的返回值和所有乘加；
- `kda_load_word()` / `kda_page_index()` 的 word/block；
- `kda_copy_words()` 的 `dst_word`、`src_relative`、`src_word`；
- `127`/`~127` 等 32 位掩码改为 `127ull`/`~127ull`。

关键表达式变为：

```c
unsigned long long dst_word =
    (unsigned long long)dst->address * 32ULL + kda_word_offset(...);
```

修复后必须重新构建 kernel 库并重启服务，因为运行中的进程已经 `dlopen` 了旧
`libcustom_dlc_perf_lib.so`：

```bash
cd /root/workspace/DLC_Custom_Kernel
./build.sh -j 16
```

该修复当前为未提交修改。不能只替换 Python 包装或重启服务；必须让新的 kernel
blob 真正进入安装目录。

### 6.2 阶段 B：warmup 后再次清零 sparse null block 0

在 `vllm-cl/vllm_cl/worker/dlc_worker.py:712`，于所有 dummy / warmup 路径结束后、
`torch.dlc.synchronize()` 之前，再次清零 null block：

```python
if hasattr(self.model_runner, "_zero_block_ids"):
    self.model_runner._zero_block_ids([0])
```

选择这个位置的原因：

- 它位于 V1 sampler dummy run 和 V2 warmup 之后，能覆盖最后一次污染写入；
- 它在服务对外 ready 之前执行，真实请求不会看到中间态；
- 它不在 `is_last_rank` 分支内，因此所有 TP/PP rank 都会执行各自 cache shard 的清理；
- 复用已有 KV block zeroer，不引入新的 kernel 或额外状态。

该修复当前为未提交修改。它只在阶段 A 修好、KDA 层恢复有限之后才可能被完整
验证；否则 layer 0 的 NaN 会继续把后续所有层污染成 NaN，掩盖 sparse 修复效果。

### 6.3 推荐的后续加固

阶段 B 的最小修复解决了线上 null block 污染，但从设计上还可以进一步消除同类问题：

1. **dummy run 不应写真实 KV cache**：将 dummy token 的 slot mapping 设为
   `PAD_SLOT_ID(-1)`，或使用专门的 scratch block，绝不写 null block。
2. **sparse kernel 做读取端防御**：在无效 top-k gather 后、QK GEMM 前，把无效
   latent 显式清零，而不是只依赖后续 softmax mask。
3. **启动不变量检查**：服务 ready 前断言所有 sparse cache 的 block 0 为全零。
4. **回归用例**：注入 NaN 到未选中的 slot，验证有效 top-k 输出不受影响；覆盖
   全无效行、部分有效行和正常行。
5. **KDA 地址宽度回归**：所有使用 `DLCTensor.address` 做单位换算的 kernel/
   helper 都增加 32/64 位静态检查与高 HBM 地址测试，至少覆盖低地址和 >16 GiB
   两组。

### 6.4 历史中标记、但不纳入本次根因的残余风险

排查过程中还标记了以下结构性问题。它们不是本次 NaN 的直接根因，发布前仍应
单独复核：

- `vllm/vllm/v1/core/single_type_kv_cache_manager.py:118-124` 只把
  `AttentionSpec` 纳入 worker 侧新 block 清零跟踪；若 KDA/Mamba state 使用
  `MambaSpec`，page 复用前是否依赖 kernel 自身清零需要单独验证。当前证据不能
  把本次故障归因于该路径，但它是 page 复用的高风险面。
- 阶段 B 的生产侧修复仍是 warmup 后补清 null block，不是最理想的源头修复；
  dummy run 应改为 `PAD_SLOT_ID(-1)` 或 scratch block，sparse kernel 也应在
  mask 前清除无效 gather 结果。
- 探针调试中曾出现 `libtorch_dlc.so` 的 `SIGFPE`（`KernelDesc::_set_dlc_tensor_impl`），
  它与额外 stride/统计探针相关，未列为 NaN 根因；复现时应保持探针零侵入。

## 7. 验证过程与结果

### 7.1 精确复现验证

**阶段 A（KDA 高地址）**

正式回归已归档为
`DLC_Custom_Kernel/pytorch_test/test_kernel/attention/kda/test_kda_state_pool_high_addr.py`。
测试使用生产形状 `PAGES=1583, HEADS=8, DIM=128`，先占用 18 GiB HBM，再断言
state 和 gather output 的 HBM 偏移都超过 16 GiB。输出预填 42，覆盖 gather 的
选中 page、mask=false page、PAD page，以及 scatter：

```bash
cd /root/workspace/DLC_Custom_Kernel
KDA_HIGH_ADDR_TEST=1 DLC_SKIP_VERSION_CHECK=1 \
  python3 -m pytest -q -s \
  pytorch_test/test_kernel/attention/kda/test_kda_state_pool_high_addr.py
```

当前修复版结果：

```text
KDA high-address gather: state=0x100000490400200 (0x490400200),
                         out=0x1000004c1e00200 (0x4c1e00200)
1 passed
```

历史临时脚本 `/tmp/test_kda_state_high_addr.py` 给出同一组位级检查，`PAGES=1583,
HEADS=8, DIM=128`：

- 普通低地址：`gather(true)`、`gather(false)`、`scatter` 全部通过；
- 占用 18 × 2 GiB dummy、state 位于 `0x100000910400200`（约 34–38.65 GiB）时：

```text
true_eq7=131072   true_eq42=0
false_eq0=131072  false_eq42=0
scatter_eq9=131072 scatter_eq0=0
```

说明修复后高地址 gather 的 true/false 两条路径和 scatter 都正确写入；而同一
脚本在修复前高地址 gather 的 `true_eq7=0`、`true_eq42=131072`，即完全没有写。
因此正式单测在旧实现下会停在 `sentinel survived: gather did not write output`。

PyTorch 侧直接跑 unittest：

```bash
DLC_SKIP_VERSION_CHECK=1 python3 pytorch/test/dlc_ops/test_kda_state_gather_scatter.py
```

结果 `Ran 11 tests ... OK (skipped=2)`。pytest 入口因仓库 `conftest.py` 与当前
pytest 9 不兼容而失败，不是算子代码问题。

**阶段 B（sparse null block）**

- 用真实 payload 重放 sparse op：干净 cache 输出 `nan=0, inf=0`；
- 对 block 0 执行 `--zero-slot 0`：NaN 消失；
- 对 block 0 注入 NaN：NaN 稳定复现。

这验证了修复目标确实是 null block 0，而不是其他偶然输入。

### 7.2 阶段切分验证

这是本次最重要的验证方式，不能用最终整条请求“看起来正常”替代中间检查：

1. **修复 A、尚未修 B**：真实请求的 layer 0–2 全部 `nan=0, inf=0`；layer 3
   的 `Glm5NextDecoderLayer` 输入有限、输出全 NaN，之后所有层被污染。首个 NaN
   从 layer 0 移到 layer 3，证明 KDA 地址修复生效，同时暴露 sparse 独立根因。
2. **修复 A + B**：重新启动无探针服务，再进行服务级验证。

如果只看到“仍然全 `!`”就回退或重复修 A，会漏掉 B；如果只做一次 clean run，
也无法区分两个根的覆盖范围。

### 7.3 服务级验证

两个修复都落地后重新启动服务：

- 8 个 `Worker_TP0..7` 全部初始化完成；
- `Application startup complete`；
- `/health` 返回 200；
- 启动过程没有再依赖硬件复位。

真实推理：

- `max_tokens=8` 的请求返回正常英文文本，不再出现 `!!!!!!!!`；
- `max_tokens=32` 的请求返回正常中文文本；
- 后续更长的请求同样以 HTTP 200 完成；
- EngineCore 日志中正常报告 running / waiting / KV cache usage，无异常重启。

### 7.4 最终状态

- 验证完成后服务已停止，`18005` 端口已释放，芯片占用进程已退出；
- 修复代码仍未提交，保留在当前工作区；
- 新增高地址单测 `1 passed`，现有 KDA gather/scatter 单测 `11 passed, 2 skipped`；
- `py_compile` 与 `git diff --check` 通过。

## 8. 本次排查的经验教训

1. **HTTP 200 不代表模型正常**：必须检查 logits、hidden states 和 token 分布。
2. **地址宽度以目标 ABI 为准**：DLC 目标上 `long`/pointer 可能是 32 位；不能用
   host Linux 的 64 位习惯推断。所有地址单位换算都要检查乘法溢出。
3. **低地址测试通过不能证明地址安全**：本次 16 GiB 以下全部正常，模型 state 在
   34 GiB 才稳定失效；必须做高 HBM 地址回归。
4. **用 sentinel 区分“算错”和“没写”**：输出先填 42/NaN，观察调用后是否改变，
   比只看 NaN 数量更快定位空写、错写和生命周期问题。
5. **先找第一个 NaN 的生产者，而不是第一个显式的 NaN 观察点**：残差流会让现象
   向下游扩散，层号本身不是根因。
6. **每修一个根因都要重新建立首个 NaN 层基线**：首个 NaN 向后移动是进展，不是
   修复失败；多根因串联是常见情况。
7. **区分 dummy run 与真实请求**：warmup 会掩盖或制造问题，是模型启动类 NaN
   的高频来源。
8. **不要相信“后面会 mask”**：mask 之前的任何 gather / GEMM 都可能已经被 NaN
   污染；必须在读取边界处理无效索引。
9. **有状态缓存要做前后快照**：不仅要看算子输出，还要看它读取的 KV/state/cache。
10. **一次 clean run 不能排除时序/地址问题**：探针关闭后问题可能暂时不复现，必须
    做确定性阈值和多次重启对照。
11. **单点脏数据足以污染整批**：405248 行 cache 中只有一行 NaN，也能让所有请求
   全量失败。
12. **硬件复位与数值错误要分开处理**：PCIe 复位解决初始化卡死，不代表 NaN 根因。
13. **精确 payload 复现是最有价值的中间产物**：它把“全模型黑盒”收缩成了一个可
    单步开关（zero slot / NaN slot）的单元测例；探针本身也要防参数遮蔽、路径
    找错、3D cache 展平等二次错误。

## 9. 模型 NaN 问题排查 SOP

以下流程面向 vLLM + DLC 模型，也可迁移到其他有状态推理后端。

### 9.1 P0：冻结现场

记录并固定：

- 模型路径、权重版本/哈希；
- 代码分支、commit、未提交 diff；
- 启动命令和全部环境变量；
- TP/PP/DP 配置、block size、max model len、量化方式；
- 编译器目标 ABI：`sizeof(long)`、`sizeof(void *)`、`DLCTensor.address` 的单位；
- 关键 state/KV cache 的 `data_ptr()`、HBM 地址范围和是否跨过 4 GiB / 16 GiB；
- 复现请求的完整 body、随机种子、temperature、tokenizer 版本；
- 服务日志、kernel 日志、profiler trace（如有）。

目标是保证任何人能在同一机器上重放同一问题。

### 9.2 P1：确定性复现

1. 使用单请求、`temperature=0`、较小的 `max_tokens`；
2. 固定 prompt 和 seed；
3. 同时记录响应 body、HTTP 状态、耗时和日志；
4. 区分 prefill、decode、warmup 三个阶段；
5. 如果初始化卡死，先 kill 所有 `VLLM::Worker`/`VLLM::EngineCore`/`vllm serve`，
   再执行 `cltechpd_clnt -s --full` 复位硬件，之后单独验证 NaN 是否仍存在。
6. 分别记录“首个 NaN 层”和“首个 NaN 张量”；每次修复后重新测一次，作为是否
   进入下一层根因的基线。

### 9.3 P2：排除外围层

依次检查：

- tokenizer / chat template；
- embedding；
- 权重是否含 NaN/Inf；
- dtype、量化 scale、位置编码；
- 采样器 logits 与 logprob；
- 显存越界、未初始化内存、race condition。

### 9.4 P3：建立 NaN 传播链

在模块级增加 in/out 探针，至少记录：

- NaN / Inf 数量；
- finite absmax；
- shape、dtype、stride、data_ptr；
- rank、step、请求 ID、warmup/real 标记；
- 该模块是否跨 TP/PP。

定位原则：

- 输入有限、输出 NaN → 当前模块是生产者；
- 输入已 NaN → 继续向上游追；
- 只有部分 rank NaN → 检查 collective、分片或 rank-specific cache；
- 所有层都 NaN → 优先找全局状态污染，而不是逐层算子。
- 首个 NaN 在某层输入 → 先查该输入的 producer、地址和缓存，不要直接归因于该层；
- 修复后首个 NaN 向更深的层移动 → 之前根因已修复，继续按新基线排查。

### 9.5 P4：精确到算子

对可疑算子抓取真实输入并落盘，构建独立重放：

- 保持 shape、stride、dtype、layout；
- 先用干净数据验证算子正常；
- 再逐项注入 NaN/Inf、无效索引、空行、极值；
- 与 native/CPU/PyTorch reference 对比；
- 做单变量实验：清零一个 slot、污染一个 slot、替换一行 top-k。
- 用 sentinel（42/NaN/特定 bit pattern）预填输出，验证 kernel 是否真的写入；
- 分别用低 HBM 地址和跨过 4 GiB / 16 GiB 的高地址重放，不能只测默认 allocator。

### 9.6 P5：审计有状态数据

对 KV cache、Mamba/KDA state、indexer cache、block table、slot mapping 做前后快照：

- 初始分配是否为零；
- null block / PAD slot 是否始终有效；
- 无效 top-k 被映射到哪里；
- 真实请求未选中的 slot 是否可能被读取；
- warmup / dummy run 是否写了真实 cache；
- block 复用前是否清零；
- 多 rank 的 cache shard 是否都满足同一不变量。
- state/KV 指针是否在高地址，地址单位换算是否在 32 位内被截断。

### 9.7 P6：审计地址宽度与高位溢出

这是本次最容易漏掉、也最值得固化的专项检查。对每个 DLC kernel/helper：

1. 用**目标 DLC 工具链**（不是 host gcc）编译最小程序，打印
   `sizeof(unsigned long)`、`sizeof(void *)` 和地址字段宽度；本次实测分别为
   4、4。地址计算代码再用显式 64 位类型兜底：
   ```c
   static_assert(sizeof(unsigned long long) == 8, "no 64-bit arithmetic");
   ```
   若 ABI 报告 `long`/pointer 只有 4 B，所有地址/偏移都必须显式使用
   `unsigned long long`/`uint64_t`。
2. 搜索危险模式：`(unsigned long)ptr`、`(unsigned long)address * N`、
   `unsigned long offset`、32 位掩码作用在地址上、`int` 承载 byte/word 地址。
3. 区分两个边界：
   - **4 GiB**：32 位 byte/pointer 本身的高位截断；
   - **16 GiB**：以 128 B 为单位的地址乘 32 后，超过 32 位 word 地址。
   本次命中的是后者，但两类都要测。
4. 测试地址扫描：低地址、4 GiB 附近、16 GiB 前后、真实模型分配地址（本次约
   34 GiB）；占用 HBM dummy 张量把 state/out 推到各区间，保持引用后再调用。
5. 对 gather/scatter 这类共享 helper 同时验证读路径、写路径和 mask=false 清零。
6. 新进程必须重新 `dlopen` 最新 kernel 库；安装 `.so` 后不重启等于没修。

### 9.8 P7：修复与防御

优先修根因，不优先在输出端 `nan_to_num`：

1. 修正生命周期：dummy 不写真实 cache，或写专用 scratch block；
2. 修正读取边界：无效索引在 gather 后立即清零；
3. 对共享 null block 增加启动断言；
4. 对关键算子增加输入有限性检查和 shape/stride 校验；
5. 修复必须覆盖所有 TP/PP rank，避免只修 rank 0。
6. 地址/ABI 问题统一改成显式 64 位，并在源层/共享 helper 一次修全，不在调用点
   做临时补丁。

### 9.9 P8：验证矩阵

至少执行以下验证：

- 精确 payload 单测：干净输入正常；故障输入可复现；清零后恢复；
- KDA/状态池：低地址与 >16 GiB 高地址的 gather true/false、scatter 全部通过；
- 空写检测：输出 sentinel 在调用后被正确覆盖，而不是保持原值；
- 启动回归：warmup 后所有 null block 仍为零；
- 分阶段回归：修 A 后确认首个 NaN 已移到 layer 3；修 B 后确认所有层 finite；
- 短请求：`max_tokens=8`，检查不再出现全 `!`；
- 长请求：覆盖 prefill、decode 和较长输出；
- 多请求：检查 block 复用、并发和前缀缓存；
- 无效 top-k 场景：全无效、部分有效、全部有效；
- 多 rank：逐 rank 检查输出和 NaN 计数；
- 性能回归：确认探针和防御逻辑未引入热路径开销。

### 9.10 P9：清理与归档

- 关闭临时探针和调试开关；
- 保存关键 payload、日志摘要和复现脚本路径；
- 记录根因、修复 commit、验证命令和结果；
- 把经验转为回归测试，避免只停留在一次性分析。

### 9.11 P10：引擎长时间 `running` 时采集芯片现场

当 vLLM 请求一直处于 `running`，但没有新增 step/token 日志，或出现 RPC
`sample_tokens` 超时时，先保留芯片现场，不要立即 kill 进程或复位硬件。
以下命令均为现场读取，不改变芯片运行状态；需要 root 权限（或对
`/sys/devices/virtual/chipltech/cltech*/reg_addr`、`reg_val` 有写权限）。
**不要附加 `peek.sh -r`**，该选项会复位状态寄存器，可能破坏现场：

```bash
cd /usr/local/chipltech/cl_tools/CsrAccess

# 全卡概览；单卡可用 -t <device_id>
./peek.sh -s status
./peek.sh -s status -t 0

# 查看当前执行到的 kernel、KIdx/KCnt 及 HBM/PCIe 状态
./peek_stuck.sh
./peek_stuck.sh -t 0

# 需要看 XYS1 时
./peek_stuck.sh -t 0 -x
```

采集要求：

1. 记录时间、请求 ID、TP rank 和当时的服务/worker PID；
2. 间隔 5～10 秒至少采样两次，并 `tee` 落盘，单次快照不能判断是否真的卡死；
3. 对 8 张卡同时记录 XYS0/XYS1 的 `State`、`PC/NPC`、`Errors`，以及
   `Kernel`、`KIdx`、`TBatch`、`KCnt`；
4. `Kernel` 列可能被截断；需要批次内核全名时可用
   `./peek_stuck.sh -t 0 --fullbatch`。

判读规则：

| 观察 | 判断与下一步 |
|---|---|
| `State=Running`，两次采样的 PC 或 KCnt 持续变化 | 芯片仍在推进，不是硬件死锁；优先查慢 kernel、collective 等待或 host 侧编排 |
| `State=Running`，同一 kernel 的 PC/KCnt 长时间不变 | 优先怀疑该 kernel 死循环、错误参数或该 rank 的 collective 卡死 |
| `State=Halted`，但引擎仍无进展 | 芯片当前空闲，优先查 worker/RPC、host 调度、进程阻塞或上层 collective，不要先复位 |
| 仅一个 rank 的 kernel/PC 与其余 rank 不同 | 查 rank-local 输入、分支、内存和该 rank 的通信状态 |
| `HBM`/`PCIe` 为 `FAIL` 或存在 `Errors` | 先按硬件/链路问题处理，不要继续做模型数值归因 |

确认现场后，再按 P1 的流程停止 `vllm serve`、`VLLM::EngineCore` 和全部
`VLLM::Worker`，并执行 `cltechpd_clnt -s --full` 复位硬件。复位前保存的
`peek` 输出应附到日志或问题单中，用于区分“kernel 未完成”和“芯片已空闲但
host 侧卡住”。

## 10. 快速分诊表

| 观察到的现象 | 优先怀疑 |
|---|---|
| 启动正常、HTTP 200、输出全 `!` | logits / hidden states NaN |
| 所有层、所有 token 都 NaN | 全局缓存、embedding、collective 或启动 warmup |
| 低地址单测全过，整模型稳定 NaN | 高 HBM 地址、32 位地址/乘法溢出、allocator 残留 |
| sentinel 调用后完全未改变 | kernel 未写、写错地址或 launch/dispatch 路径错误 |
| 仅某个 head/切片是 qNaN，其余是旧位型 | 未初始化 buffer 或错地址，而非统一数值溢出 |
| 修完一层后首个 NaN 移到更深层 | 存在第二个独立根因；按新层号继续排查 |
| 只有某一 rank NaN | TP/PP 分片、all-reduce、rank-local cache |
| 只有部分 token 或部分行 NaN | 索引、mask、padding、无效 slot |
| prefill 正常，decode 变 NaN | KV cache 内容、block 复用、decode slot mapping |
| 干净输入正常，复用 cache 后 NaN | null block / PAD slot / cache 清零生命周期 |
| 第一次请求正常，第二次 NaN | 有状态写入或 block 释放/复用错误 |
| NaN 在 mask 后仍存在 | NaN 已在 QK/GEMM 等前置计算中传播 |
| 复位硬件后仍稳定 NaN | 软件/数据路径问题，不是 PCIe 或驱动状态 |
| 引擎长时间 `running`，无新 token/step | 按 9.11 用 `peek.sh` / `peek_stuck.sh` 采现场；先区分芯片推进、kernel 卡死和 host 卡住 |
| 芯片 `Running` 但同一 kernel 的 PC/KCnt 不变 | kernel 或 rank-local collective 疑似卡死 |
| 芯片全部 `Halted` 但 worker 无进展 | 优先查 host/RPC/调度阻塞，不是芯片指令死循环 |

## 11. 证据与代码索引

| 类型 | 路径 |
|---|---|
| 阶段 A 修复：KDA 64 位地址运算 | `DLC_Custom_Kernel/dlc_kernels/attention/kda/kda_state_pool.h:5-121` |
| 阶段 A 正式高地址单测 | `DLC_Custom_Kernel/pytorch_test/test_kernel/attention/kda/test_kda_state_pool_high_addr.py` |
| 阶段 A 历史临时脚本 | `/tmp/test_kda_state_high_addr.py` |
| 阶段 A PyTorch 单测 | `pytorch/test/dlc_ops/test_kda_state_gather_scatter.py` |
| 阶段 B 修复位置 | `vllm-cl/vllm_cl/worker/dlc_worker.py:712` |
| V1 dummy warmup | `vllm-cl/vllm_cl/worker/dlc_worker.py:687-713` |
| KV cache 分配 | `vllm-cl/vllm_cl/worker/dlc_model_runner.py:7399` |
| dummy slot mapping | `vllm-cl/vllm_cl/worker/dlc_model_runner.py:3989-4007` |
| KVBlockZeroer warmup | `vllm/vllm/model_executor/warmup/kernel_warmup.py:169-174` |
| block 0 清零实现 | `vllm/vllm/v1/worker/utils.py:252-255` |
| sparse kernel 无效 top-k 映射 | `DLC_Custom_Kernel/dlc_kernels/attention/sparse_attention/sparse_attention_kv_cache/sparse_attention_kv_cache_bf16.cpp:2040-2042` |
| cache 写入 kernel | `DLC_Custom_Kernel/dlc_kernels/kvcache/concat_and_cache_mla_bf16.c:65-79` |
| 阶段 B 精确复现脚本 | `/tmp/glm5_sparse_exact_repro.py` |
| 完整 cache dump | `/tmp/glm5_sparse_aten14.pt.pid3196566.call0.post.pt` |
| 写 cache 探针样本 | `/tmp/glm5_cache_write_probe.pt.pid3215584.call0.pt` ~ `call10.pt` |
| 早期 NaN 追踪日志 | `/tmp/glm5_nan_trace.log` |

## 12. 一句话总结

**地址换算必须按目标 ABI 使用足够位宽，并有高 HBM 回归；有状态 null block 必须
在 warmup 后仍保持全零，且任何无效索引都不能在 mask 前被当作合法 slot 读取。**
