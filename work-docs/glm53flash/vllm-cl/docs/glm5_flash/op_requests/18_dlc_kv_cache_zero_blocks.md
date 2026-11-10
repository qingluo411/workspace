<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22）：可复用已有算子（本条撤回，不提交）
     （09-22 先按「需扩展」拟；再按「芯片只做 512B 对齐算子」复核 GLM5 实际页几何后改为「可复用」） -->
<!-- 已有算子覆盖核查（2026-09-21/09-22，DLC 实测）：已被 dlc_vllm_zero_block_ids_ 覆盖
     证据 1：GLM5 全部 5 类页（4096~524288 B/块）都是 512 B 的整数倍，官方参数形态下多段调用 bit-exact（exact_match=True）
     证据 2：kernel 以 ALIGN128(page_size_el) 同时作写入长度与 block 步长 → 前提是「每块字节数 % 512 == 0」且 page_size_el = 块字节数/4；GLM5 全部页类满足
     证据 3：同一次调用内若混用不同页大小的 segment，会越界清错相邻块 → 接入时按「每块字节数」分组调用
     建议：撤回新增算子提名；改为插件接入（int64 地址表 + 按页字节分组 + 段等长连续），不需要算子组动作 -->

# [Operator Request] dlc_kv_cache_zero_blocks（KV cache 物理块清零）

## 需求类型
其他

**需求分类：可复用已有算子（本条撤回，不提交）**
- 现有算子：`torch.dlc_vllm_zero_block_ids_(seg_addrs, block_ids, n_blocks, n_segs, page_size_el, num_blocks_total)`
  （kernel 名 `custom_zero_block_ids`，源码 `DLC_Custom_Kernel/dlc_kernels/vllm/zero_block_ids.cpp`）。
  `seg_addrs` 为 1D int64 绝对字节地址表、`block_ids` 为 int32、`page_size_el` 以 **int32 字**计；无 CPU fallback。
- **GLM-5.3-Flash 页几何的 512B 对齐核查（真机 spec：TP8、`--block-size 256`、`index_kpool=4`、`kv_lora_rank=512`、`mla_use_nope=true`）**：

  | 页类（零清入口） | 每块形状 | dtype | 每块字节 | ÷512 | `page_size_el`(int32) | ÷128 |
  |---|---|---|---|---|---|---|
  | MLA full（11 层 DSA 全量 cache） | `[1, 256, 512]`（256 token × 512 latent） | bf16 | 262,144 | 512 | 65,536 | 512 |
  | KPool pool（11 层 indexer 池 cache） | `[64, 128]`，DLC 行宽补到 256 | bf16 | 32,768 | 64 | 8,192 | 64 |
  | KPool tail（11 层环形 tail） | `[2, 4, 128]`，行宽补到 256 | bf16 | 4,096 | 8 | 1,024 | 8 |
  | KDA conv state（34 层短卷积状态） | `[3, 3072]` | bf16 | 18,432 | 36 | 4,608 | 36 |
  | KDA recurrent state（34 层递推状态） | `[8, 128, 128]` | fp32 | 524,288 | 1024 | 131,072 | 1024 |

  （34 层 KDA 的两种 state 计入 `MambaSpec`，真机日志 `mamba_page_bytes=542720` = 18432+524288；
   注意 `_entries` 是**每个页类各一个张量入口**（共 67 个），不是把同层两种 state 合成一个大页，
  因此传给 kernel 的 `page_size_el` 是上表逐行的值，而不是 542720/4。）
  （pool / tail 即使按未补齐的逻辑行宽算（16,384 B / 2,048 B）也同样是 512 B 整数倍，结论不依赖 DLC 行宽补齐的具体取值。）
- 实测（dlc:0，2026-09-22，`probes/probe_zero_block_ids_glm5_geometry.py`）：
  按上表 5 类页各建 3 个 segment（独立分配）、每段 8 块、清 `{1,3,6}`，
  **5 类页全部 `exact_match=True`**（`ALL_ALIGNED_AND_EXACT: True`），命中块为 0、相邻块逐元素未变。
- 另测（混合页反例）：同一次调用里混用 32,768 B 与 4,096 B 的 segment → 两段都不匹配。原因是 kernel 对
  全部 segment 只用一个 `page_size_el`：按大页算时小页那段会写到张量尾部之外（并清错相邻分配），
  按小页算时大页那段只被部分清零。**所以接入时必须按「每块字节数」分组调用，同组内 segment 等长、地址连续。**
- 结论：**现有算子已覆盖本需求，GLM5 不存在非 512B 对齐的页**，不需要扩展、不需要算子组动作。
  撤回原提名的新算子 `dlc_kv_cache_zero_blocks`，改为 vllm-cl 插件侧接入（见「期望 API 与函数签名」的封装）。
- 备查（非本次请求）：kernel 的对齐要求来自 `hbm_sizewords = ALIGN128(page_size_el)` / block 步长
  `= ALIGN128(page_size_el)/32`；只有当「每块字节数 % 512 == 0」时 `ALIGN128` 才是恒等变换。
  若将来出现非 512B 整数倍的页（本模型没有），才需要请算子组支持按字节寻址。
- 复现脚本：`probes/probe_zero_block_ids_glm5_geometry.py`（GLM5 页几何，含混合页反例）、
  `probes/probe_zero_block_ids.py`、`probes/probe_zero_block_ids_page_alignment.py`。
- 本次调整（2026-09-22）：需求类型「扩展已有算子能力」→「其他（复用已有算子）」；优先级 P1→P2；
  算子名称改为复用 `torch.dlc_vllm_zero_block_ids_`（原提名 `dlc_kv_cache_zero_blocks` 撤回）；
  **本条撤回，不提交给算子组**。

## 优先级
P2

## 目标版本与期望时间
不提交给算子组、不需要排期。由 vllm-cl 插件侧在 GLM-5.3-Flash 适配 Phase 2 内接入
`torch.dlc_vllm_zero_block_ids_`（当前 `zero_block_ids_pytorch` 的 `zero_()` 调用处于注释状态，
接入后需恢复「新分配块必须清零」的正确性）。

## 算子名称
复用 `torch.dlc_vllm_zero_block_ids_`（原提名 `dlc_kv_cache_zero_blocks` 撤回，不新增）

## 背景与功能说明
使用场景：vLLM V1 的 `KVBlockZeroer` 在每步把调度器新分配给请求的 KV cache **物理块**清零，防止上一个请求残留在块内的脏 K/V 被新请求误读。上游判定 `needs_kv_cache_zeroing = has_mamba_layers or has_mixed_precision_kv_cache`（`vllm/v1/kv_cache_interface.py:1383`）——**GLM-5.3-Flash 有 34 层 KDA mamba 状态，必然命中**，因此这是默认调度路径上的每步操作，不是可选的优化项。

当前问题：DLC 用纯 PyTorch 实现替代上游 Triton kernel（`vllm_cl/patch/patch_kv_block_zeroer.py:193` 的 `zero_block_ids_pytorch`），性能不可接受：

```python
for kv_cache, block_dim, ratio in self._entries:          # 67 个张量入口
    block_stride = kv_cache.stride(block_dim)
    outer_dims = [dim for dim in range(block_dim) if kv_cache.stride(dim) > block_stride]
    outer_indices = list(iprod(*outer_ranges)) if outer_ranges else [()]
    for bid in block_ids:
        for outer_index in outer_indices:
            index = [slice(None)] * kv_cache.ndim
            ...
            index[block_dim] = slice(block_start, block_start + ratio)
            kv_cache[tuple(index)].zero_()                # 慢在这里（DLC 上 view/布局转换很慢）
```

三个叠加的开销：
1. **Python 双重/三重循环**：层数（34 KDA + 11 MLA + 11 KPool pool + 11 tail = 67 个张量入口）× block 数 × `outer_dims` 组合。
2. **切片 view 触发布局转换**：`kv_cache[tuple(index)]` 在 DLC 上会走 view/layout 转换路径（shape-stride 合法性检查 + 可能的物化重排），这正是不用 Triton 后最慢的一环；`[block_dim] = slice(...)` 这种混合了整维 slice 与标量的索引尤其容易命中该路径。
3. **每层独立 launch**：每个 entry 每个 block 一次写零，无法合并。

另外需要说明的现状：当前工作区版本里写零调用被注释掉了（`patch_kv_block_zeroer.py:214` 的 `# kv_cache[tuple(index)].zero_()`），即**清零实际未执行**。这属于用正确性换性能的临时绕过；改用 `dlc_vllm_zero_block_ids_` 后即可恢复该语义（这才是本条目保留在文档里的原因）。

受影响张量（异构布局；这也是不能与计算类算子合并的原因）：
- MLA cache：`[blocks, 1, 256, 512]` BF16
- KPool pool cache：`[blocks, 64, 128]` BF16（物理行宽补到 256）；KPool tail cache：`[blocks, 2, 4, 128]` BF16
- KDA conv state：`[blocks, 3, 3072]` BF16；KDA recurrent state：`[blocks, 8, 128, 128]` FP32

目标功能：用现有 `dlc_vllm_zero_block_ids_` 单（或少量）kernel 完成「按物理块 id 列表，对一组异构 cache 张量写零」，不在 Python 侧构造任何 view。
非目标：不做 KV 数据写入（那是 `concat_and_cache_mla`）；不改变调度器的块分配策略；不做跨卡/PD 传输；不要求算子组做任何改动。

## 期望 API 与函数签名
**算子组侧：无新增、无参数变更**，直接复用现有签名：

```python
# 已存在（DLC torch）
torch.dlc_vllm_zero_block_ids_(
    seg_addrs: torch.Tensor,   # [n_segs] int64，绝对字节地址（DLC 无 uint64 storage）
    block_ids: torch.Tensor,   # [n_blocks] int32
    n_blocks: int,             # = block_ids.numel()
    n_segs: int,               # = seg_addrs.numel()
    page_size_el: int,         # 每个 block 的 int32 字数 = 块字节数 // 4
    num_blocks_total: int,     # kernel 内未使用（仅 API 兼容）
) -> None
```

**插件侧封装（vllm-cl 自行实现）**：

```python
def zero_block_ids_dlc(self, block_ids: list[int]) -> None:
    """按 (page_size_el, seg_block_stride) 分组，每组一次 launch。"""
    if not block_ids or not self._meta:
        return
    bid = torch.tensor(block_ids, dtype=torch.int32, device=self.device)
    for (page_el, stride_el), group in self._groups.items():
        addrs = torch.tensor([g.addr for g in group], dtype=torch.int64, device=self.device)
        torch.dlc_vllm_zero_block_ids_(addrs, bid, bid.numel(), len(group), page_el, 0)
```

## 参数与返回值
输入（沿用现有算子的口径）：
- `seg_addrs`：`[E]`，int64，每个 segment 的**绝对字节地址**；`E` 为按页大小分组后同组内的 segment 数。
- `block_ids`：`[B]`，int32，本步新分配的 block id（GLM5 上 `B` 为个位数到几十），必须支持批量。
- `page_size_el`：int32，**每块需清零的 int32 字数**；GLM5 取值见上表（1024 / 4608 / 8192 / 65536 / 131072）。
- `num_blocks_total`：现有 kernel **未使用**（源码 `(void)p_num_blocks_total;`），传 0 或任意值均可。

约束（实测得出，接入时必须满足）：
1. 每块字节数（= `page_size_el * 4`）必须是 **512 B 的整数倍**，否则 kernel 会按 `ALIGN128` 放大跨度、清错块并越过张量尾部；
2. 同一次调用内所有 segment 的 `page_size_el` 相同（kernel 只有一个 `page_size_el`），且每个 segment 在显存中是
   `[num_blocks, page_size_el]` 的**等长、连续** int32 布局；不同页大小的张量必须拆成多次调用；
3. `seg_addrs` 必须是 int64 一维表（DLC 不支持 `uint64` storage）；
4. 无 CPU fallback，地址表必须在设备上。

返回值：无（原位写设备内存）。边界：`B == 0` 或 `E == 0` 时 kernel 内 `n_blocks == 0` 直接返回（安全空操作）。

## 功能语义
```text
for (page_size_el, seg_group) in groups:              # 插件侧按页大小分组
    for b in block_ids:
        for s in seg_group:                            # kernel 内两次循环：block × segment
            addr = seg_addrs[s] + b * (ALIGN128(page_size_el) * 4)   # 因 512B 对齐，ALIGN128 为恒等
            addr[0 : page_size_el] = 0                 # 整块写零（含 ratio>1 时的全部物理 page）
```

要点：
1. **不在 Python 侧构造任何 view / 切片**；地址计算全部在 kernel 内完成，这是解决「view 布局转换慢」的关键。
2. 一次 launch 覆盖同组内全部 entry 与全部 block id，避免逐层逐块 launch。
3. 清零是幂等的纯写操作，不读取原内容，因此不需要考虑原数据的 NaN/Inf。
4. 不同层张量地址互不重叠（DLC 每层独立分配），不需要上游的 overlay/`max(page_size)` 逻辑。
5. 对齐前提见上：`ALIGN128(page_size_el) == page_size_el` 当且仅当块字节数是 512 的整数倍。

in-place / alias：直接写设备内存，不返回张量；不与其他算子产生 alias 语义。
空 / 零长度：`B == 0` 或 `E == 0` 为空操作。
确定性：无随机，纯写零。

## 支持范围与约束
支持的 dtype：`block_ids` int32、`seg_addrs` int64（DLC 无 `uint64` storage）。
支持的 shape：`B` 动态（1~数百）；同组内 `E` 动态（GLM5 单组最大 34，即 34 层 KDA conv state）。
数值范围：`page_size_el` 见上表，最大 131,072 int32 字（KDA recurrent），远小于 kernel 的 VMEM 上限
（`page_size_el < 1024 * 3072`）；`block_ids` < `num_blocks`（GLM5 TP8 下约 1500~2000）。
layout / stride：每层张量都是独立的连续分配、块维在 dim 0（KDA state）或解析出的 block 维，
`stride(block_dim) * element_size * ratio` 即上表「每块字节」，均为 512B 整数倍。
device：DLC 单卡。
执行模式：按现有 `_zero_block_ids` 每步 eager 调用即可；若后续在 CUDA graph 内使用，
需要在插件接入时单独验证（本条不要求算子组支持）。
其他限制：
1. 不得依赖 Triton / tilelang / aiter。
2. 清零范围必须覆盖整个逻辑块（`ratio > 1` 时包含全部物理 page）。
3. Mamba/KDA state 张量与 MLA/KPool cache 张量都要支持（同一路径，按页大小分组）。

## 精度要求
Golden / reference：`vllm_cl/patch/patch_kv_block_zeroer.py` 的 `zero_block_ids_pytorch`（恢复 `zero_()` 后的语义）。
输入数据分布：不在真实 cache 上验证，构造与真实布局一致的小型张量池（页几何照抄上表，块数 × 8）。
- 预填非零哨兵值（含 NaN/Inf/极大值），调用后逐元素断言为 0；
- 覆盖 `ratio > 1`（logical block 跨多个物理 page）的 span。
样本数量：每类页 ≥ 100 组，另加 5 类页混合调用（按页大小分组后）的组合用例。
比较 dtype：按各张量原 dtype 逐元素比较。
atol / rtol：**要求 bit-exact 全 0**（纯写操作，不允许近似）；实测 5 类页均 `exact_match=True`。
必须逐元素一致的场景：
1. `block_ids` 内的块全部为 0；
2. `block_ids` 之外的块**完全不被修改**（block 步长算错就会误伤相邻块，必须用「相邻块预填哨兵」验证）；
3. 相邻层的张量（独立分配）不被误写。
NaN / Inf / 极值处理：原值即使是 NaN/Inf 也必须被写成 0（不做算术，直接写）。
累加与中间计算精度：不涉及。

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

具体测试用例、shape、数据范围和预期结果（接入回归用，算子本体已实测）：
1. 功能：按上表 5 类页各建 segment，`B ∈ {1, 8, 64}`；覆盖 67 entry 全量分组。
2. 误伤校验：命中块与相邻块预填不同哨兵，清零后相邻块必须逐元素不变（已通过）。
3. 边界：`B=0`、`E=0`、`block_ids` 全部重复、`block_ids` 含最大值边界块。
4. 分组正确性：同一步内同时存在 5 类页时，必须发出 5 次调用（每次一个 `page_size_el`），结果与逐类调用一致。
5. 异常：`block_ids` 越界、页字节数非 512 倍数时插件侧主动报错（不交给 kernel）。
6. 性能：比较对象为当前 PyTorch 实现（Python 循环 + `zero_()` view）。目标：
   - GLM-5.3-Flash 实际规模（67 entry、B=16）单次清零 **< 0.5 ms**，相对当前实现加速 ≥ 20×；
   - 消除 `zero_()` 的 view/layout 转换路径（可通过 profiler 确认无 layout conversion / copy kernel）。

## 验证环境与依赖版本
DLC 硬件：8 卡，设备名 `/dev/cltech0-7`（TP=8，无 PP）
vLLM commit：`f5361fb09f`
vLLM DLC（vllm-cl）commit：`a14fc5b`
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：`VLLM_USE_DLC_COL_MAJOR_MATMUL=1`
模型：GLM-5.3-Flash-Mix（AWQ 4-bit，`linear_num_heads=64`、`linear_head_dim=128`、`short_conv_kernel_size=4`、`gate_lower_bound=-5.0`）

## 验收标准
- [x] 页几何核对：GLM5 全部 5 类页每块字节数均为 512 B 的整数倍（`probe_zero_block_ids_glm5_geometry.py` → `ALL_ALIGNED_AND_EXACT: True`）
- [ ] 插件侧完成接入：int64 地址表、按页字节分组、同组等长连续、`page_size_el` 非 512B 倍数时报错
- [ ] 恢复 `zero_block_ids_pytorch` 的 `zero_()` 语义后，GLM-5.3-Flash 在 DLC 上 prefill/decode 数值与 PyTorch reference 一致
- [ ] 清零前后相邻块逐元素不变（哨兵用例）
- [ ] 性能对比：67 entry + B=16 单次 < 0.5 ms（或用 profiler 确认无 view/layout 转换与逐层 launch）
（本条不提交给算子组，因此无需「算子组确认对齐/几何约束」项。）

## Reference、日志与附件
Reference 实现：
- DLC 侧：`vllm_cl/patch/patch_kv_block_zeroer.py:193`（`zero_block_ids_pytorch`）、`:66`（`init_meta_pytorch`，地址/块维解析逻辑）。
- 上游 Triton 版：`vllm/v1/worker/utils.py:105`（`KVBlockZeroer`）、`:227`（`zero_block_ids`）、`_zero_kv_blocks_kernel`。
- 现有 DLC kernel：`DLC_Custom_Kernel/dlc_kernels/vllm/zero_block_ids.cpp`（`hbm_sizewords = ALIGN128(page_size_el)`）。
- 调用点：`vllm_cl/worker/dlc_model_runner.py:1102-1117`、`dlc_model_runner_v2.py` 的对应 `_zero_block_ids`；
  上游判定入口 `vllm/v1/kv_cache_interface.py:1383`（`needs_kv_cache_zeroing`）。
为什么本需求不需要新算子：
- 现有算子已在 GLM5 的真实页几何下 bit-exact（含多段调用）；
- 本算子作用在「块分配生命周期」上，输入是调度器给出的 block id 列表与异构张量的原始地址表，无法作为任一计算 kernel 的 prologue/epilogue 融合（计算 kernel 不感知块分配），也覆盖不到 KPool tail / KDA conv state 这类不被单一计算算子触及的张量；
- 上游本身也是把它作为独立的 zeroing kernel 实现，而非融进 attention kernel。
页几何与真机日志来源：
- 模型配置 `/mnt/jfs/models/GLM-5.3-Flash-Mix/config.json`（`index_kpool=4`、`index_head_dim=128`、`kv_lora_rank=512`、`mla_use_nope=true`）；
- 真机日志 `vllm-cl/glm53_flash.log:6424`（`kinds={'MambaSpec': 34, 'MLAAttentionSpec': 22, 'KpoolTailSpec': 11}`、`mamba_page_bytes=[542720, 542720]`）、
  `:6446`（`allocate_mamba_cache: ... page_size_bytes=542720 shapes=((3, 3072), (8, 128, 128))`）；
- 页大小计算：`vllm_cl/patch/patch_kv_cache_utils.py:195-239`、`vllm_cl/kpool_cache_layout.py:58`（`native_kpool_page_bytes`）、
  `vllm_cl/patch/models/glm5_indexer.py:89-118`（pool / tail spec 定义）。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_联调问题与修复记录.md`（其中记录了 `get_kv_cache_block_dim` 缺失导致 `_init_kv_zero_meta` 报错的问题，属同一子系统的接口缺口）。
