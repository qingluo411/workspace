# GLM-5.3-Flash vLLM-CL 联调问题与修复记录

> 记录日期：2026-09-21
> 范围：GLM-5.3-Flash 在 DLC 自研芯片 + vLLM/vLLM-CL 上的首次真机联调（从改启动脚本到 KV cache 分配成功）。
> 状态：KV cache 分配阶段已通过，服务可启动并接受请求；但首个推理请求仍会 hang，尚未完成端到端前向验证。
> 说明：本文只记录**真实启动/推理过程中暴露的问题**及处置，不含仅静态验证的阻断点（见 `GLM-5.3-Flash_vLLM-CL_阻断点2-7.md`）。

## 0. 环境与前置调整

| 项 | 内容 |
|---|---|
| 模型 | `/mnt/jfs/models/GLM-5.3-Flash-Mix`（AWQ 4-bit，120 个 safetensors，约 184 GB） |
| 设备 | `/dev/cltech0-7`，8 卡；前 4 卡内 / 后 4 卡内通信快，跨前后四卡慢 |
| 精度 | `--dtype bfloat16`（权重 int4，激活 BF16） |
| 启动脚本 | `docs/glm5_flash/start_glm5_flash_vllm_cl.sh` |

**并行策略决策：TP=8（不做 PP）。**

- 放弃 PP=2 × TP=4：GLM-5.3-Flash 是 45 层（奇数），PP 切分边界容易和 MTP/共享 expert 打架；且现有 KDA/KPool/MLA 补丁只验证过 TP-only 路径，PP 会引入跨阶段激活通信和层映射的额外变量。
- 接受 TP 的 8 卡 all-reduce 开销，换取最成熟、已验证的代码路径。后续做性能优化时再评估 PP=2 × TP=4。

脚本改动：`MODEL_PATH` → `/mnt/jfs/models/GLM-5.3-Flash-Mix`，`TP_SIZE` 默认 8，显式 `--gpu-memory-utilization 0.85`。

---

## 1. 多模态视觉塔 profile 阶段触发 Triton kernel

**现象**

服务启动到 profile run 阶段，Worker_TP0 报错，调用栈落在 `vllm/models/common/ops/fused_qk_rmsnorm.py`。

**根因**

多模态视觉塔的 dummy 前向走到上游 Triton 版 `fused_q_kv_rmsnorm`（视觉塔 Q/K RMSNorm）。DLC 无 Triton，插件此前没有该算子的对应补丁。

**修复**

按现有补丁风格新增 native 替换：`vllm_cl/patch/patch_fused_qk_rmsnorm.py`（并在 `vllm_cl/patch/__init__.py`、`patch_import_manifest.yaml` 注册）。

**验证**：服务可越过 profile 阶段。

---

## 2. AWQ MoE `qzeros` 布局与 DLC kernel 不匹配

**现象**

AWQ MoE 权重后处理阶段 shape mismatch：传给 `torch.dlc_vllm_fused_moe_w4a16_awq_col_major` 的 `qzeros` 布局不符合 kernel 期望。

**根因**

DLC fused MoE kernel 期望 `qzeros` 为 **colpack 布局**（`[num_groups, ceil(packed_cols/128)*128]`），而插件的 `process_weights_after_loading` 错误地复用了 `qweight` 的 rowpack + col-major 转换路径。kernel 期望 `w13_qzeros: [288, 32, 512]`，实际传入的是 rowpack 后的布局。

**修复**（`vllm_cl/ops/awq_marlin.py`）

- 新增 `process_qzeros_tensor()`，改用专用算子 `torch.awq_uint4_to_dlc_uint4_colpack`：只把最后一维 pad 到 128 的倍数，**不做** rowpack / col-major。
- `w13_qzeros` / `w2_qzeros` 改走该路径；`qweight` / `scales` 保持原 rowpack + col-major 不变。
- 清理两个不再使用的中间变量。

**验证**（DLC 设备实测）

- GLM5 尺寸（hidden 4096 / intermediate 2048 / group 128）下 `w13_qzeros → (32, 512)`、`w2_qzeros → (16, 512)`，与 kernel 期望逐维一致。
- 非 128 对齐尾块 pad 正确（`520 → 640`）。
- `tests/test_glm5_awq_weight_mapping.py` 7/7 通过（并把测试内模型路径同步改到 `/mnt/jfs`）。

**后续观察点**：AWQ MoE 数值正确性（`swiglu_limit=10.0` clamp 是否生效）。

---

## 3. OOM（一）：KV cache 分组与容量核算打架

**现象**

分配 mamba cache 时 OOM，`Tried to allocate 674 MiB`，且 `GPU KV cache size` 只有 `45,409 tokens`（= 177 × 256），远低于可用显存应有的量。

**根因**

DLC 插件自己的容量核算与上游 GLM5 专用分组互相矛盾：

1. 上游 `get_kv_cache_groups`（`kv_cache_utils.py`）识别 GLM5Next 混合拓扑，把 34 个 KDA 层 merge 成**一个**组，并给 spec 打上 `page_size_padded = MLA 页 = 256 KiB` 的共享页假设；
2. 但 DLC worker 实际是**每层独立分配** `num_blocks × page_size`，34 层只被计一次、实际要分 34 份；
3. 真实 KDA 页（约 0.518 MiB）> MLA 页（0.25 MiB），共享页假设本就不成立。

结果：`num_blocks` 被高估约 6.5 倍，核算与分配脱节 → OOM。

**定位方法**

离线复现 spec → grouping → `num_blocks` 的计算链，并补 CPU 单测复现"分组 → 核算 → 分配"三者一致性。

**修复**

在 DLC 的 KV cache 配置入口加 GLM5 专用分池分组（`vllm_cl/patch/patch_kv_cache_utils.py`）：把 KDA 合并组拆成**每层一个独立 spec**（`page_size_padded=None`，按真实页计费），彻底绕开上游的共享页假设。

**验证**

- 分组结果：1 组 full MLA + 1 组 KPool index + 1 组 KPool tail + **34 个 KDA 层各自独立 Mamba 组**。
- `num_blocks = 1509`，`1509 × 20.734 MiB ≈ 30.55 GiB ≤ 30.56 GiB` 可用——核算与 worker 实际分配完全一致。
- `tests/test_glm5_kv_grouping_dlc.py` 3/3 通过（期间修正测试 mock：`KpoolTailSpec` 需 `head_size_v=0`、`block_size=4`，与 `vllm_cl/patch/models/glm5_indexer.py` 的真实构造一致）。
- 相邻回归 33/33、扩大到 kda/mamba/kv_cache/prefix 相关 60/60 通过。
- 连带路径确认：`_init_input_batch` 按组独立算块表宽度，与"每组几层"无关。

---

## 4. OOM（二）：mamba 页被 pad 到 MLA 整倍数，浪费约 6 GiB

**现象**

修复分组后仍 OOM，`num_blocks` 依旧显示 177；进一步看日志发现 KDA 真实页被 pad 到 `786432 B`（0.75 MiB = 3 × MLA 页），而 `GPU KV cache size` 仍是旧值。

**根因**

上游 `_align_hybrid_block_size` 为 NVIDIA 的"attn 页 ≥ mamba 页"共享页布局，把每个 KDA 页从真实 `542720 B` pad 到 `786432 B`（+42%）。DLC 每层是独立的 raw byte-page tensor，只是按 `get_state_shape()/get_state_dtype()` 解包，padded 尾部字节**从不被访问**。34 层 × 771 块 × 232960 B ≈ **5.7 GiB 纯浪费**，正好造成缺口。

**诊断手段（本轮最有效的一步）**

不再靠离线推算，直接在真实代码路径加诊断日志（spec 类型统计、`page_size` / `page_size_padded`、分组结果、每层实际分配字节数、`allocated/free` 变化），用真实启动日志定位。确认真实 spec 形状为 `(3,3072)` conv + `(8,128,128)` recurrent，真实页 `542720 B`。

**修复**（`vllm_cl/platform.py`）

重写 `update_block_size_for_backend`：解析 backend 首选 block size，但**跳过 `_align_hybrid_block_size` 的 mamba padding**（DLC 不需要该约束）。使 `page_size_padded=None`，核算与分配全链路统一用真实页 `542720 B`。

> 已验证前提：`bind_kv_cache` 只读真实页的 `542720 B`，padded 尾部从不访问；`page_size_bytes` 属性在 `page_size_padded is None` 时返回真实值，核算/分配口径一致。

**同时尝试但无效的方向（记录以免重复踩坑）**

- `PYTORCH_DLC_ALLOC_CONF=expandable_segments:True`：**DLC 平台不支持**（`DLCAllocatorConfig.h` 明确警告），此路不通。脚本里目前仍保留该 export，可视为无效项。
- 降 `gpu_memory_utilization` 到 0.85：能缓解但治标，根因仍是 padding 浪费。

**验证**

- 补丁生效：`padded=None`、真实页 `542720 B`、`num_blocks=1172`、单层 mamba 606.6 MiB。
- **34 层 mamba 全部分配完成，不再 OOM**；`GPU KV cache size` 升到 `148,762 tokens`（此前 45,409）。

---

## 5. KV block zeroer 缺接口：`get_kv_cache_block_dim`

**现象**

KV 分配完成后进入 `_init_kv_zero_meta`（`vllm_cl/patch/patch_kv_block_zeroer.py`），调用 `PyTorchMLASparseBackend.get_kv_cache_block_dim` 报 `AttributeError`。

**根因**

`get_kv_cache_block_dim` 是 vllm-cl 自己扩展的接口（上游没有），zeroer 会对所有 `FullAttentionSpec` 的 backend 调用它，但稀疏 MLA、KPool index、KPool tail 三个 backend 只实现了 `get_kv_cache_shape`，漏了该方法。

**修复**

- `vllm_cl/attention/sparse_mla.py`：给 `PyTorchMLASparseBackend` 加 `get_kv_cache_block_dim`，返回 `0`（shape 为 `(num_blocks, block_size, head_size)`，block 维在 dim 0）。
- `vllm_cl/patch/patch_mla_indexer.py`：给 `DeepseekV32IndexerBackend` / `KpoolTailBackend` 补同一方法（shape 分别为 `[B, N, C]`、`[B, 2, N, C]`，block 维同样在 dim 0），与各自的 `get_kv_cache_shape` 对齐。

**验证**：服务可继续启动到下一步。

---

## 6. `find_longest_cache_hit` 返回契约不一致

**现象**

推理请求进入调度后 EngineCore 崩溃，`allocate_slots` 之前的 `find_longest_cache_hit` 解包报错。

**根因**

本轮适配把基类 `SingleTypeKVCacheManager.find_longest_cache_hit` 的返回注解改成了 `tuple[tuple[list, ...], int]`（2 元：`computed_blocks, hit_length`），但子类 `MambaManager`、`SlidingWindowManager` 的实现仍只返回 `computed_blocks`（1 元），coordinator 按 2 元解包即失败。

GLM-5.3-Flash 会同时触发这两类 manager：34 层 KDA → `MambaManager`；KPool tail（`KpoolTailSpec` 继承 `SlidingWindowSpec`）→ `SlidingWindowManager`。

**修复**（`vllm/v1/core/single_type_kv_cache_manager.py`）

两个 manager 的返回注解与返回值都改为 2 元，补 `hit_length = len(computed_blocks[0]) * block_size`，与 `FullAttentionManager` 语义一致。

> 范围控制：`RSWAManager` 继承 `FullAttentionManager`（已是 2 元，OK）；`ChunkedLocalAttentionManager` 同样返回 1 元，但 GLM5 不经过它，本轮按最小修复原则未改，作为已知的同类风险记录在案。

---

## 7. `MambaManager.get_num_blocks_to_allocate` 缺参数

**现象**

`MambaManager.get_num_blocks_to_allocate() got multiple values for argument 'apply_admission_cap'`。

**根因**

基类签名为 8 个位置参数 + 1 个关键字参数，`MambaManager` 覆写时漏了 `num_local_computed_tokens`，调用方按位置传参就把 `num_tokens_main_model` 和 `apply_admission_cap` 都塞给了 `apply_admission_cap`。

**修复**（`vllm/v1/core/single_type_kv_cache_manager.py`）

- 覆写签名补上 `num_local_computed_tokens`；
- `super().__init__(...)` 的位置参数同步补上该参数。

> 已确认只有基类和 `MambaManager` 覆写了该方法，其余 manager 直接走基类，无同类问题；基类内部本就 `del num_local_computed_tokens`。

---

## 8. `torch.searchsorted` 在 DLC backend 未注册

**现象**

请求下发后 worker 报算子未实现：`torch.searchsorted` 在 DLC dispatcher 中找不到 kernel。调用点是 `vllm_cl/kpool_worker_slots.py:57`（KPool tail slot 计算）。

**根因**

DLC dispatcher 未注册 `searchsorted` / `bucketize`。

**修复**（`vllm_cl/kpool_worker_slots.py`）

把 `searchsorted` 放到 CPU 上算再搬回 DLC：

```python
batch_rows = (
    torch.searchsorted(starts.cpu(), tokens.cpu(), right=True)
    .to(positions.device)
    - 1
)
```

**性能对比**：CPU 往返约 4 ms，broadcast 比较实现约 29 ms；CPU 方案更快且语义完全一致。该逻辑另落到 `vllm_cl/_searchsorted_utils.py`。

---

## 9. Mamba state copy funcs 形态不对（单数 → 复数 dict）

**现象**

`MambaCopyBuffers.create` 按 `copy_funcs[mamba_type]` 索引时报错。

**根因**

DLC runner 直接调 `self.model.get_mamba_state_copy_func()`，返回的是 **tuple of 4 funcs**；而 `mamba_utils.MambaCopyBuffers.create` / `preprocess_mamba` / `postprocess_mamba` 期望的是 **按 mamba_type 索引的 dict**（`{GDN_ATTN: (conv, temporal)}`）。上游 vLLM runner 用的是复数形式 `get_mamba_state_copy_funcs(mamba_types)`（内部构造 dict + 校验），DLC runner 缺这一步。

> 补充事实：`Glm5NextLinearAttention.mamba_type = GDN_ATTN`（不是 KDA）；KDA 有 conv + recurrent 两个 state，`kda_state_copy_func` 返回 2 个 funcs，满足 `0 < len(copy_funcs) <= len(shapes)`。

**修复**（`vllm_cl/worker/dlc_model_runner.py`）

- 新增 `_get_mamba_state_copy_funcs()`：`get_mamba_groups` → 收集 `mamba_type` → `model.get_mamba_state_copy_funcs(...)` → `validate_mamba_state_copy_funcs(...)`，并缓存结果；
- 三处调用（`MambaCopyBuffers.create`、`preprocess_mamba`、`postprocess_mamba`）由单数 `self.model.get_mamba_state_copy_func()` 改为 `self._get_mamba_state_copy_funcs()`。
- 同步改 `vllm_cl/worker/dlc_model_runner_v2.py`。

---

## 10. DLC 不支持 `uint64` buffer dtype

**现象**

`DLCStorageImpl.h:105 (get_dlc_dtype)` 报错，定位到 `MambaCopyBuffers.create` 里创建 GPU buffer 时 dtype 不被支持。

**根因**

`MambaCopyBuffers.create` 内部为 `src_ptrs` / `dst_ptrs` 硬编码 `dtype=torch.uint64`，而 DLC 存储层不认识 `uint64`（`int32` / `int64` 可用）。

**修复**（`vllm_cl/worker/dlc_model_runner.py`）

在 `_make_buffer` 入口拦截：`torch.uint64 → torch.int64`（本平台小端，指针值语义不变）。

---

## 11. 运维性问题：nohup 启动日志 0 字节且进程即退

**现象**

用 `nohup` 后台启动后日志文件 0 字节，进程随即消失；前台跑却能正常起。

**处置**

改用 `setsid` 后台启动确保进程组与日志重定向正常。属启动方式问题，非代码缺陷。

---

## 12. 待解决 / 未验证

1. **首个推理请求仍 hang（最高优先级）**。当前日志停在：服务已起、`Running: 1 reqs`、`GPU KV cache usage 2.9%`，EngineCore 反复 `Poller timed out`，且 60 秒内 `No available shared memory broadcast block found`。尚不确定是文本侧 KDA/MLA 首次前向耗时过长，还是 `shm_broadcast` 相关卡死，需继续定位。
2. **诊断日志需清理**：`patch_kv_cache_utils.py` 里为定位加的 spec/分组诊断打印属于临时手段，定位结束后应移除或降级为 debug。
3. **脚本中无效项**：`PYTORCH_DLC_ALLOC_CONF=expandable_segments:True` 在 DLC 平台不生效，确认无副作用后可删。
4. **已知同类风险未修**：`ChunkedLocalAttentionManager.get_num_blocks_to_allocate` / `find_longest_cache_hit` 契约与 `MambaManager` 同类，GLM5 当前不触发，但若被其它模型路径复用会同样报错。
5. **数值验收未做**：AWQ MoE（`swiglu_limit=10.0`）、KDA/MLA 首次前向的数值正确性，以及 `supports_fp8=False` 与默认 checkpoint 的匹配，都还没在真机上验证。

---

## 13. align 模式 Mamba 状态迁移走 Triton：长 prompt 第二个 chunk 必崩

**现象**

任何**跨 mamba block 边界**的请求（`mamba_cache_mode == "align"` 下
`mamba_block_size = block_size = 256`，即序列长度跨过 256 的整数倍）在
`execute_model` 一开始就抛：

```
dlc_model_runner.py:4021  mamba_utils.preprocess_mamba(
vllm/v1/worker/mamba_utils.py:1538  do_mamba_copy_block(copy_bufs)
vllm/v1/worker/mamba_utils.py:1379  batch_memcpy(...)
vllm/v1/worker/mamba_utils.py:667   batch_memcpy_kernel[grid](...)
TypeError: 'function' object is not subscriptable
```

复现：向 8000 端口发一条 **1211 token** 的 prompt（`/tmp/req_long.json`）。
第一个 chunk（1024 token）无拷贝所以能过；第二个 chunk（`num_computed_tokens=1024`、
`num_scheduled_tokens=187`）触发 `prev_state_idx=3 != curr_state_idx=4` → 状态迁移 →
崩溃。5 token prompt + 128 输出（=133 token，未跨 256 边界）不会触发，所以基础
benchmark 一直没暴露它。

**根因**

vLLM 的 align 模式在跨块时要把请求的 conv/recurrent state 从旧块列搬到新块列。
GPU 路径是 Triton kernel `batch_memcpy_kernel`；非 GPU/无 Triton 时插件既没有兜底、
DLC worker 也没传 `align_ctx`（fused precopy 路径同样依赖 Triton），于是落到
`do_mamba_copy_block` → `batch_memcpy_kernel[grid]`。DLC 上 `triton_utils.importing`
检测到 active driver 数为 0 会把 `triton` 换成占位实现（`triton.jit` 变 identity
装饰器），kernel 名只是普通函数，`[grid]` 自然抛 `TypeError`。

**处置**

新增 `vllm_cl/patch/patch_mamba_state_copy.py`（在 `patch/__init__.py` 注册）：
仅当 `HAS_TRITON == False` 时替换 `mamba_utils.collect_mamba_copy_meta` /
`mamba_utils.do_mamba_copy_block`。保持上游的 staging 协议（同样的提前返回条件与
`copy_bufs.offset` 计数），但记录**张量切片**而不是裸地址，用 eager `copy_` 完成迁移；
同一块内（spec decode 接受 token 后才可能出现的重叠拷贝）先 clone 源。要求每个
state 是独立连续池（拆池方案已满足），非连续则直接报错而不是静默搬错字节。

代价：一次完整跨块迁移 = 34 层 × 2 state = 68 次 `copy_` / 18.5 MB，实测
**dlc:0 上 1.79 ms**（对比 decode 每步 ~369 ms），可以忽略。

配套：`vllm_cl/patch/models/kpool_indexer.py` 的
`only a complete pool's last token may have a pool slot` 现在会把出问题的
请求号、token、position、pool/tail slot 打进报错信息（只在失败路径上做 D2H）。

测试：`tests/test_mamba_state_copy_native.py`（新增，7 例）覆盖安装替换、地址→切片
解析、重叠 clone、非连续拒绝、多 state 迁移与步边界重置；
`tests/test_kpool_metadata_native.py` 补上 AST fixture 缺失的
`searchsorted_right_minus_one`（4 例由报错转为通过）。

---

## 附：本轮变更文件

**vllm（`/root/workspace/vllm`）**

- `vllm/v1/core/single_type_kv_cache_manager.py`：`MambaManager` / `SlidingWindowManager` 补 2 元返回契约与 `num_local_computed_tokens` 参数
- `vllm/models/glm5next/nvidia/kda.py`：conv1d 权重跳过 DLC col-major 后处理（`_dlc_skip_col_major_weight_processing`）

**vllm-cl（`/root/workspace/vllm-cl`）**

- `vllm_cl/patch/patch_fused_qk_rmsnorm.py`（新增）、`patch/__init__.py`、`patch_import_manifest.yaml`：视觉塔 RMSNorm native 替换
- `vllm_cl/ops/awq_marlin.py`：`qzeros` colpack 路径
- `vllm_cl/patch/patch_kv_cache_utils.py`：GLM5 专用分池分组
- `vllm_cl/platform.py`：跳过 mamba 页 padding
- `vllm_cl/attention/sparse_mla.py`、`vllm_cl/patch/patch_mla_indexer.py`：`get_kv_cache_block_dim`
- `vllm_cl/kpool_worker_slots.py`、`vllm_cl/_searchsorted_utils.py`：`searchsorted` CPU 回退
- `vllm_cl/worker/dlc_model_runner.py`、`dlc_model_runner_v2.py`：mamba state copy funcs 复数化、`uint64 → int64`
- `docs/glm5_flash/start_glm5_flash_vllm_cl.sh`：模型路径、TP=8、util 0.85
- `tests/test_glm5_kv_grouping_dlc.py`（新增）、`tests/test_kda_native.py`、`tests/test_glm5_awq_weight_mapping.py`
- `vllm_cl/patch/patch_mamba_state_copy.py`（新增）、`patch/__init__.py`：align 模式 Mamba 状态迁移的 eager 兜底
- `vllm_cl/patch/models/kpool_indexer.py`：KPool 池槽不变式报错带上现场（请求号/position/slot）
- `tests/test_mamba_state_copy_native.py`（新增）、`tests/test_kpool_metadata_native.py`
