## 新项目：DLC 异步 D2H 机制与 vLLM 异步调度适配

- **时间**：2026-10-20 至 2026-11-02
- **角色**：框架路径分析 / 性能评估（研究整理，非 PyTorch 侧代码作者）
- **技术栈**：`PyTorch 2.5.0` DLC fork、`vLLM` V1、`vllm-cl`、DLC runtime（C/C++）、Python
- **涉及模型**：`GLM-5.3-Flash-Mix`（45 层、hidden 4096、34 个 KDA 层、11 个 sparse MLA 层；本文分析针对其 DLC 推理路径）

### 项目背景

DLC 后端的 `tensor.to("cpu", non_blocking=True)` 长期并不真正异步。旧实现虽然把拷贝放在独立 stream 上，但 PyTorch 侧仍走同步 staging：先把 DLC 物理 padded storage 拷到 pinned CPU 临时区，再由 CPU `unpack`，最后 `copy_` 到用户输出。DLC runtime 还会对非 pinned 或非 512B 对齐的 D2H 强制 `isAsync=false`，所以 worker 主线程实际会卡在 D2H 的提交与完成上。

vLLM 的异步调度本来已经用 `AsyncScheduler`、独立 copy stream 和 event 做了“先调度下一批、后取上一批输出”的框架，但 DLC 后端的同步 D2H 把这层重叠能力吃掉了。目标分支 `fix/async-d2h-layout-kernel-1028` 的改动，正是把 DLC D2H 变成真正的异步 DMA，使采样 token 回 CPU 不再阻塞下一批模型计算。

本文的目标不是重复实现该功能，而是回答三个问题：

1. PyTorch 侧具体如何把 D2H 做成真异步；
2. vLLM/vllm-cl 的实际 D2H 点位在哪里，下一批输入如何在没有 CPU token 的情况下准备好；
3. 这项优化在 DLC 生产路径上可能带来多大收益，以及哪些路径仍然不受益。

### 职责

- **负责**：阅读目标分支 commit 与 DLC runtime 实现，梳理 D2H 调用链、生命周期和 O2/O3 开关；定位 vLLM/vllm-cl 的实际 D2H 点位；拆解 `_prepare_inputs` 中上一 step GPU token 到下一 step 输入的依赖；整理收益估算和验证方案。
- **参与**：与已有性能文档、发布验证数据做交叉核对；只做代码与数据分析，不把 PyTorch commit 的实现工作归为个人产出。
- **未参与**：PyTorch commit 的实际编码；DLC runtime 的驱动/DMA 实现；目标分支的重新构建和端到端 A/B 性能测试。

### 关键行动

#### 1. 框架接入与机制研究

目标分支相对 merge base 主要有两个 commit：

- `313fc47c31`：实现异步 D2H fast path、512B padded pinned 输出、device-side layout kernel。
- `20feb31a48`：把异步 D2H 和 padded pinned 输出收紧到 `O3`；`O3` 仍为默认；把原先只在 O3 的 H2D copy-kernel 路径下沉到 O2；移除 device tensor keepalive；目标 tip 同时包含 `input_format(src)` 的 source-stride 修复。

**一次 `.to("cpu", non_blocking=True)` 的实际调用链：**

```text
to("cpu", non_blocking=True)
  -> _to_copy()
  -> 分配 padded pinned CPU output
  -> r.copy_(self, non_blocking=True)
  -> copy_through_cpu()
  -> handle_d2h_copy()
  -> try_device_layout_async_d2h()
  -> device compact/layout kernels
  -> synMemCopyAsync(D2H, padded_bytes)
  -> return（不等待 DMA 完成）
```

**1.1 输出先变成 padded pinned storage**

`pytorch/aten/src/ATen/native/TensorConversions.cpp:343` 只在 DLC、`non_blocking=True` 且 O3 时把 CPU 输出标记为 pinned。  
`TensorConversions.cpp:354` 的 `empty_padded_dlc_d2h_output()` 会：

- 计算 `padded_bytes = ceil(logical_bytes / 512) * 512`；
- 分配 `at::empty({padded_numel}, pinned_memory=true)`；
- 用 `set_` 把逻辑 shape/stride 设回原来的 `self.sizes()/self.strides()`。

因此 vLLM 的 `int32 [N,1]` 采样结果逻辑上仍是 `[N,1]`，但底层 storage 至少有 512B。DMA 按 padded bytes 写，不会越过分配边界；用户侧 `.tolist()` 只看到逻辑元素。

**1.2 只有满足条件才进入异步 fast path**

`pytorch/aten/src/ATen/native/dlc/Copy_cpu.h:787` 的 `can_use_device_layout_async_d2h()` 要求：

- source 是 DLC、连续、`storage_offset()==0`；
- destination 是 CPU、连续、非 view、唯一持有；
- destination 由 DLC CachingHostAllocator 分配；
- `dst.storage().nbytes() == aligned_d2h_bytes(src)`；
- dtype 在支持列表中；
- 未设置 `DLC_DISABLE_DEVICE_D2H_LAYOUT=1`。

不满足时，`Copy_cpu.h:958` 回退到旧的同步 `dlc_tensor_to_cpu_storage()` + CPU unpack。

**1.3 把 CPU unpack 搬到芯片上**

`Copy_cpu.h:818` 的 `try_device_layout_async_d2h()` 做三件事：

1. `flat_view = src.reshape(-1)`，检查 `is_contiguous_dlc(flat_view)`；
2. 如果物理布局不是一行，启动 `custom_strided_get*` kernel 压缩成一行；
3. 8/16/64-bit 再用 `custom_convert_to_cpu_layout_*` 转成 CPU 自然字节序；32-bit 已同序，不需要转换。

大于 16384 个元素时分块 compact，避免 strided-get 的输入窗口问题。`Copy_cpu.h:853` 使用 `kernel.input_format(src)` 而不是 `flat_view`，因为 `reshape` 只是 metadata view，不能消除物理 padding；vLLM 的 `[N,1]` 采样结果正好需要原始 strides。

**1.4 最终只发一次对齐的 async DMA**

`Copy_cpu.h:920` 调用 `custom_device_memcpy(host, device, padded_bytes, D2H, stream, true)`。  
`Copy_cpu.h:150` 在 `non_blocking=true` 时走 `synMemCopyAsync`，不会调用 `dlcStreamSynchronize`；旧路径传 `non_blocking=false`，是 `dlcMemcpyAsync` 后立即同步。

**1.5 runtime 为什么以前强制同步**

`DLCSynapse/synapse_core/synapse_state.cpp:1812` 计算：

```cpp
useRegularMemOrD2hMisaligned =
    (!isPinnedHugepageMem && !isDmabufMem && !dmaMappedIndex) ||
    (dmaSize != logicalSize && !toDevice);
```

只要 host 不是 pinned/DMA-mapped，或 D2H 的 `dmaSize != logicalSize`，runtime 就在 `:1820` 设置 `isAsync=false`，分配临时 host buffer，同步完成后再拷回用户 buffer。  
新路径把 destination 变成 pinned，并把 DMA 大小变成 512B 整数倍，才真正满足 runtime 的 async 条件。

**1.6 生命周期保护**

函数返回后，`src`、`flat`、`dma_source` 这些局部 Tensor 可能被释放，但 DMA 还在队列里。`Copy_cpu.h:930` 对它们逐个调用 `DLCCachingAllocator::recordStream()`；host destination 则在 `:941` 通过 `CachingHostAllocator_recordEvent()` 记录到同一个 stream，防止 pinned buffer 被 allocator 提前回收。

因此 PyTorch 侧只保证“提交异步、完成有序”；调用方必须像 vLLM 一样等 event 或同步 stream，不能 `.to()` 后立刻读数据。

**1.7 O3 开关**

`pytorch/c10/dlc/DLCConstants.h:16` 统一解析 `DLC_SYN_COPY_ASYNC`：

- 未设置或未知值：O3；
- `O0/O1/O2`：不会启用异步 D2H；
- O3：开启 padded pinned 输出和异步 D2H。

该环境变量在每个进程内静态读取一次，必须在进程启动前设置。

#### 2. 精度对齐与数据正确性

这项工作不是模型数值精度对齐，而是 D2H 拷贝语义的正确性验证。

**2.1 逻辑 shape/stride 与物理 storage 分离**

输出只改变底层 storage 大小，不改变用户可见 shape/stride。padding 只存在于 storage 尾部，逻辑元素数量和顺序不变。

**2.2 布局转换的正确性**

旧路径由 CPU `unpack_dlc_transfer()` 完成 DLC padded layout 到普通 CPU layout 的转换。新路径把同样的职责交给 device kernel：

- 32-bit：DLC 物理元素顺序与 CPU 自然顺序一致，只做 compact；
- 8/16/64-bit：compact 后再做 `custom_convert_to_cpu_layout_*`；
- 大输入：分块 compact，避免单次 strided-get 窗口越界；
- 没有对应 compact kernel 的 dtype/形状：回退同步路径。

**2.3 已知风险点**

- `flat_view` 不能代替 `src` 作为 strided-get 的输入 descriptor；否则 padded `[N,1]/[N,4]` 可能拿到错误的一行 metadata，甚至挂住设备队列。
- 目标分支 tip 已用 `kernel.input_format(src)` 修复。
- destination 必须是唯一持有的 padded storage；共享大 buffer 的 slice/view 会被拒绝，防止对齐 DMA 覆盖逻辑尾部之外的数据。

**2.4 测试覆盖**

`pytorch/test/dlc_async_d2h_copy.py` 覆盖：

- `.to()` 返回后 event 仍 pending；
- padded pinned output；
- 连续、非连续、切片、转置；
- 不同 dtype/尾长度；
- 大输入分块；
- 多次 queued copy 的生命周期；
- source/flatten/layout temporary 提前释放；
- stream 依赖、Python dispatch、fallback error、H2D、DLC-to-DLC、autograd。

其中 `test/dlc_async_d2h_copy.py:92` 的 `check_copy_is_deferred_to_event()` 直接验证“提交后返回，event 稍后才完成”。

#### 3. vLLM/vllm-cl 异步调度适配分析

**3.1 实际生产路径是 vllm-cl 的 `DLCModelRunner`**

`vllm-cl/vllm_cl/worker/dlc_worker.py:301` 在非 V2 时选择 `DLCModelRunnerV1`。上游 `vllm/v1/worker/gpu_model_runner.py` 只是同构参考，不是 DLC 实际执行路径。

**3.2 AsyncScheduler 本身不做 D2H**

`vllm/v1/core/sched/async_scheduler.py:39` 只维护 `num_output_placeholders`，让下一批可以在上一批真实 token 还没回 CPU 时先调度。D2H 发生在 worker 的 `AsyncGPUModelRunnerOutput`。

**3.3 实际 D2H 点位**

`vllm-cl/vllm_cl/worker/dlc_model_runner.py:276` 记录 default stream，`:277` 切到 `async_output_copy_stream`，`:279` 发起：

```python
self.sampled_token_ids_cpu = self._sampled_token_ids.to(
    "cpu", non_blocking=True
)
```

`:282` 到 `:286` 再处理 logprobs，`:287` 记录 `async_copy_ready_event`。`:294` 的 `get_output()` 在异步输出线程里等 event，`:301` 之后才 `.tolist()` 或做 rejection parse。

**3.4 下一批输入不需要 CPU token**

async decode 时，`_bookkeeping_sync` 把 `sampler_output.sampled_token_ids` 直接留在 GPU，并赋给 `input_batch.prev_sampled_token_ids`，见 `dlc_model_runner.py:3553` 和 `:3557`。  
下一 step 的 `_prepare_inputs()` 在 `:1867` 开始，先准备 CPU 侧调度元数据，再调用 `_prepare_input_ids()`：

- 没有 `prev_sampled_token_ids`：prefill/新请求，`input_ids.cpu` 整体 H2D；
- 有 `prev_sampled_token_ids`：对 batch 中与上一批共有的 decode 请求，直接把 GPU 上的 sampled token 填入 `input_ids.gpu`；
- batch 顺序不变时用直接 slice copy，见 `:1778`；
- batch 重排或混合新请求时用 `scatter_`，见 `:1792`；
- speculative decoding 再 scatter draft tokens，见 `:1800`。

因此普通 decode 的下一步模型输入是 `prev_sampled_token_ids`，不是 CPU 上的 `.tolist()` 结果。CPU 只提供 positions、query_start_loc、slot mapping、seq_len 等调度元数据。

**3.5 主线程与异步输出线程**

- worker 主线程：`sample_tokens()` 返回 `AsyncGPUModelRunnerOutput` 后，通过 `vllm/v1/executor/multiproc_executor.py:1000` 把对象放进 `async_output_queue`，随后回到 `:1027` 的 worker RPC 循环，继续执行下一批 `execute_model`。
- 异步输出线程：在 `:1010` 的 `async_output_busy_loop()` 中调用 `:981` 的 `get_output()`，等待 event，做 `.tolist()`/parse，再把 `ModelRunnerOutput` 放进 response MQ。
- EngineCore：V1 且 `pp=1` 时 `max_concurrent_batches=2`，见 `vllm/config/vllm.py:589`；`step_with_batch_queue()` 可以先提交下一批，等 batch queue 满后才在 `vllm/v1/engine/core.py:743` 的 `future.result()` 上等上一批输出。

**3.6 可重叠与不可重叠的操作**

可以重叠：

- 下一 step 的 `_prepare_inputs()`：positions、query_start_loc、slot mapping、block table、pinned H2D metadata；
- 下一 step 的 GPU 前向：KDA/MLA、MoE、allreduce、sampling；
- 异步输出线程里的 event 等待、`.tolist()`、输出解析；
- EngineCore 的下一批调度、请求管理和 aborts 处理；
- 多批在飞时，更后面批次的计算。

不能或只能部分重叠：

- 当前 step 的 `scheduler.update_from_output()` 仍要等 D2H 完成，因为要用真实 sampled token 更新请求状态和 KV cache；
- 结构化输出需要上一步真实 token 计算 grammar bitmask 时，下一步 sampling 会被推迟，见 `vllm/v1/engine/core.py:721`；
- repetition penalty、bad words 等依赖 output token ids 的 logits processor，可能通过 `vllm/v1/worker/gpu_input_batch.py:1042` 等 event；
- speculative decoding 的 draft token/valid count 预分配 `copy_` 路径不满足新 fast path；
- 没有 async scheduling、`enforce_eager=True`、O2 或 `DLC_DISABLE_DEVICE_D2H_LAYOUT=1` 时，优化基本不生效。

#### 4. 性能分析

**4.1 旧路径的真实成本**

旧路径的成本不只是 DMA 字节数，而是：

- runtime 对非 pinned/非对齐 D2H 强制同步；
- 临时 host buffer 分配；
- DMA 完成等待；
- CPU layout unpack；
- 再从 staging `copy_` 到用户输出。

对小 token id 来说，固定开销占主导，所以即使 payload 只有 512B，也可能出现明显 host stall。

**4.2 新路径的收益来源**

- 去掉了 CPU unpack 和额外 `copy_`；
- 用 device kernel 完成 compact/layout；
- 用 padded pinned output 满足 runtime 的 async 条件；
- D2H 提交后 worker 主线程即可继续下一批；
- event 等待转移到异步输出线程，和下一批计算重叠。

**4.3 量级判断**

- 规划文档曾估算采样 D2H 落到 pinned buffer 的收益约 **5 ms/step**，来源：`work-docs/glm53flash/vllm-cl/docs/glm5_flash/下一步工作计划_2026-10-10.md:86`。
- 最近一次发布验证窗口里，40 次 `.tolist()` 合计 **0.2 ms**，4 次 copy-back 合计 **0.427 ms**，无 >0.5 ms 阻塞 D2H，来源：`work-docs/glm53flash/vllm-cl/docs/glm5_flash/GLM-5.3-Flash_发布前功能与性能验证_2026-10-22.md:136`。
- 当前干净单请求 decode 约 **96.9 ms/step**，graph 单请求 decode gap 约 **67.64 ms**。

按这些步长折算：

- 1 ms/step 约提升 1.0% 到 1.5%；
- 5 ms/step 约提升 5.2% 到 7.4%。

因此合理预期是：**低并发、无 logprobs 时可能是亚毫秒到 1 ms/step；开 logprobs、speculative decoding 或 host stall 明显时可能到数 ms/step。它主要优化 TPOT/worker 前推，不会解决当前约 90 ms 的 collective/barrier 等待。**

### 量化结果

| 指标 | 数值 | 条件与来源 | 状态 |
|---|---:|---|---|
| 异步 D2H 事件语义 | `.to()` 返回后 event 仍 pending | `pytorch/test/dlc_async_d2h_copy.py:92` | commit 测试已覆盖，本次未重跑 |
| 规划估算收益 | 约 5 ms/step | 采样 D2H 落 pinned buffer；`下一步工作计划_2026-10-10.md:86` | 估算，非本轮 A/B |
| 最近发布验证 `.tolist()` | 40 次合计 0.2 ms | 发布验证窗口；`发布前功能与性能验证_2026-10-22.md:136` | 已有测量 |
| 最近发布验证 copy-back | 4 次合计 0.427 ms | 同一验证窗口 | 已有测量 |
| 干净单请求 decode | 约 96.9 ms/step | GLM-5.3-Flash-Mix，生产配置附近 | 历史基线 |
| graph 单请求 decode gap | 约 67.64 ms | 49 prompt + 96 out，FULL_DECODE_ONLY | 历史基线 |
| 1 ms/step 相对收益 | 约 1.0%–1.5% | 以 96.9/67.64 ms 为分母 | 估算 |
| 5 ms/step 相对收益 | 约 5.2%–7.4% | 同上 | 估算 |

**结论：目前没有目标分支上的同条件 O2/O3 A/B 数据。** 当前 venv 的 `torch.version.git_version` 仍是 merge base `9049914...`，不是目标分支 tip `20feb31a...`；要量化必须重新构建并安装目标 PyTorch。

### 面试深挖点

1. **为什么旧的 `non_blocking=True` 实际上不是异步？**  
   关键在 DLC runtime：非 pinned/非 512B 对齐 D2H 会设置 `isAsync=false`；旧 PyTorch 路径又走 CPU staging + unpack，并没有真正满足异步条件。

2. **为什么必须把输出 padding 到 512B？**  
   runtime 把 `dmaSize != logicalSize` 视为 misaligned D2H，会分配临时 buffer 并强制同步。padding 后 DMA 大小与分配边界都对齐，才能保持 `isAsync=true`。

3. **为什么还要 device-side layout kernel？**  
   DLC 物理 storage 可能按 padded row 存放，不能直接把逻辑 tensor 当连续 buffer 拷给 CPU；必须先 compact，并对 8/16/64-bit 做 CPU 字节序转换。

4. **异步后如何防止 source/host buffer 被提前释放？**  
   设备侧用 `DLCCachingAllocator::recordStream()`，host 侧用 `CachingHostAllocator_recordEvent()`；调用方再用 event 等待完成。

5. **为什么 vLLM 下一批模型输入不需要 CPU token？**  
   普通 decode 的下一步输入就是上一步 sampled token，而它已经在 GPU 的 `prev_sampled_token_ids` 中；`_prepare_input_ids()` 在 GPU 上直接 copy/scatter 到 `input_ids.gpu`，CPU 只提供调度元数据。

### 口述基线（待核验）

#### 当前叙述

GLM/DLC 推理里，异步调度的目标本来是让 worker 在上一批采样结果还没回 CPU 时继续跑下一批。但旧 DLC 后端即使传 `non_blocking=True`，D2H 仍会因为 host 非 pinned、尺寸未 512B 对齐而走 runtime 的同步 staging，worker 主线程实际上被卡住。

我做的不是 PyTorch commit 的编码，而是把这条链路完整拆开：PyTorch 侧通过 padded pinned output、device-side compact/layout kernel、`synMemCopyAsync`，以及 `recordStream`/host event 生命周期保护，把 D2H 变成真正提交后返回；vLLM-cl 侧实际点位在 `DLCModelRunner` 的 `AsyncGPUModelRunnerOutput`，不是上游 `gpu_model_runner`。普通 decode 的下一批输入直接取 GPU 上的 `prev_sampled_token_ids`，CPU 上的 token 回读只为 response、请求状态和部分采样约束服务。

最难的判断点有两个：一是 512B 对齐和 pinned 是 runtime 是否保持异步的硬条件；二是 vLLM 的 `_prepare_inputs` 在 async scheduling 下必须能区分“CPU 已知的 prefill/新请求 token”和“GPU 上的上一 step sampled token”。这两点对上之后，才能确认主线程可以与 D2H 重叠的是下一批 prepare、H2D metadata、模型前向和 sampling。

已有数据能支持的结论是：这项改动方向正确，但当前 GLM 路径已经把长阻塞 D2H 清掉，实际收益更可能是亚毫秒到数毫秒/step；要给出可信的百分比，必须在目标 PyTorch 分支上做 O2/O3、logprobs、spec decode、不同并发度的同条件 A/B。

#### 个人职责与协作边界

- **我负责**：代码路径研究、机制拆解、vLLM/vllm-cl 点位映射、收益估算和文档整理。
- **与同事共同完成**：需要与 PyTorch commit 作者确认目标分支构建状态；与 runtime/性能同事确认多 stream DMA 与 compute 的真实并行能力。
- **其他团队负责**：PyTorch DLC 后端实现、DLC runtime/driver、vllm-cl 生产接入与最终性能验收。
- **当前不能扩张的表述**：
  - 不能声称“已实测提升 5 ms/step”；
  - 不能声称“所有 D2H 都已异步化”，预分配 slice 的 `copy_`、structured output、spec draft metadata 仍有同步点；
  - 不能把上游 `vllm` 的 `gpu_model_runner` 当成 DLC 生产路径；
  - 不能声称 D2H 已从关键路径消失，`scheduler.update_from_output()` 仍然要等 event。

#### 待核验字段

| 字段 | 当前状态 | 关闭条件 |
|---|---|---|
| 模型与框架版本 | GLM-5.3-Flash-Mix；PyTorch 2.5.0 DLC；目标分支 tip `20feb31a48`；当前 venv 仍是 merge base `9049914` | 重建目标 PyTorch，记录 `torch.version.git_version` |
| 代表性改动 | 已核对 `313fc47c31` 与 `20feb31a48` 的源码路径 | 在目标构建上运行 `test/dlc_async_d2h_copy.py` |
| 精度结果 | 未做模型精度评测；拷贝正确性由 commit 测试覆盖 | 固定输入、参考输出、容差和覆盖范围 |
| 性能口径 | 只有历史/规划数据，无本轮同条件 A/B | O2 vs O3，同模型、同输入输出长度、同并发、同 dtype、同预热/重复次数 |
| 对外披露边界 | 机制可讲；具体数字需确认是否可用于对外材料 | 与项目负责人确认发布口径 |

### 附录：关键代码索引

| 文件 | 作用 |
|---|---|
| `pytorch/aten/src/ATen/native/TensorConversions.cpp:343` | DLC O3 padded pinned 输出入口 |
| `pytorch/aten/src/ATen/native/TensorConversions.cpp:354` | 512B padded pinned allocation |
| `pytorch/aten/src/ATen/native/dlc/Copy_cpu.h:787` | 异步 D2H eligibility |
| `pytorch/aten/src/ATen/native/dlc/Copy_cpu.h:818` | device compaction/layout + DMA fast path |
| `pytorch/aten/src/ATen/native/dlc/Copy_cpu.h:930` | device allocation stream lifetime |
| `pytorch/aten/src/ATen/native/dlc/Copy_cpu.h:941` | pinned host allocation event |
| `pytorch/c10/dlc/DLCConstants.h:16` | `DLC_SYN_COPY_ASYNC` O0–O3 解析 |
| `DLCSynapse/synapse_core/synapse_state.cpp:1812` | runtime 判定 misaligned/unpinned D2H 并强制同步 |
| `vllm-cl/vllm_cl/worker/dlc_model_runner.py:276` | 实际 D2H copy stream 起点 |
| `vllm-cl/vllm_cl/worker/dlc_model_runner.py:279` | sampled token id `.to("cpu", non_blocking=True)` |
| `vllm-cl/vllm_cl/worker/dlc_model_runner.py:295` | 异步输出线程等待 event |
| `vllm-cl/vllm_cl/worker/dlc_model_runner.py:3557` | 上一 step sampled token 缓存在 GPU |
| `vllm-cl/vllm_cl/worker/dlc_model_runner.py:1693` | `_prepare_input_ids()` |
| `vllm-cl/vllm_cl/worker/dlc_model_runner.py:1778` | 下一 step 直接 copy `prev_sampled_token_ids` |
| `vllm-cl/vllm_cl/worker/dlc_model_runner.py:1792` | batch 重排时 scatter sampled token |
| `vllm/config/vllm.py:589` | V1 async scheduling 的 `max_concurrent_batches=2` |
| `vllm/v1/executor/multiproc_executor.py:1010` | worker 异步输出线程 |
