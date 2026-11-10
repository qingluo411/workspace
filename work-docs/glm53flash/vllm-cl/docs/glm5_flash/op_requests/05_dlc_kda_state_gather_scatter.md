<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；重写日期：2026-09-29（只保留算子相关内容） -->
<!-- 需求类型：新增算子；优先级：P0 -->

> **状态（2026-10-10）：已交付（kernel `#8021` / pytorch `#2154`）并接入插件，本 issue 关闭。**
> `vllm_cl/patch/patch_kda.py` 的 `gather_initial_states_dlc` / `scatter_states_dlc` 默认走
> `torch.dlc_kda_state_gather/_scatter`（`DLC_KDA_STATE_POOL_OP=kernel`，`fast`=页表、`native`=旧组合实现回退）。
> 真机 34 层 gather+scatter **0.76 ms/步**（S=1；页表实现 5.6 ms），单次 gather 0.022 ms / scatter 0.013 ms，
> 无 host D2H、可 `DLCGraph` 捕获。验收：`probes/probe_kda_state_gather_scatter_op.py`（dlc:0/1/2/3 全绿）、
> `probes/probe_patch_kda_state_pool_dispatch.py`（kernel/fast/native 三模式 + 前置条件回退）。
>
> **历史状态（2026-09-30）：插件侧已用「页表」Python 实现顶上，本算子降级为可选。**
> 算子同事给出的 `kda_state_pool_dlc.py` 已收编为 `vllm_cl/patch/kda_state_pool_dlc.py`
> （开关 `DLC_KDA_STATE_POOL_OP`，默认 `fast`），在连续 `[P, page_len]` 页表上做
> `index_select` / 高级索引整行写回，页内 `narrow` 出递归窗口。真机（P=1168、H=8、Dv=Dk=128、
> fp32、拆池 dense）：gather 0.48→0.16 ms、scatter 1.99→0.09 ms（每层每次），
> 打包页几何下分别是 2176 ms→0.29 ms、11931 ms→0.16 ms；且不再有 `torch.any`/`unique`
> 的 D2H 同步。**因此本 issue 不再阻断接入**，交付后可再省 ~11 ms/步（34 层 × 0.32 ms）
> 与 68 次 kernel 下发。下面的语义要求（尤其 `-1` 跳过、bit-exact、golden reference）不变。

# [Operator Request] dlc_kda_state_gather_scatter（KDA 递归状态按页 gather/scatter）

## 需求类型
新增算子

## 优先级
P0

## 目标版本与期望时间
vLLM DLC / 与 KDA 系列（#01/#02/#03/#04）同批交付；期望里程碑待算子组排期确认

## 算子名称
`torch.dlc_kda_state_gather` 与 `torch.dlc_kda_state_scatter` —— **两个独立算子**（本 issue 一起提，
但请注册两个 aten 入口、两次 kernel 下发；文件名的 `gather_scatter` 只是打包名）：

- 两者的调用点被 #03 `dlc_kda_chunk_prefill` 隔开：gather 在 chunk 计算**前**取 `initial_state`，
  scatter 在 chunk 计算**后**把 `last_recurrent_state` 写回池
  （`vllm/models/glm5next/nvidia/kda.py:539` 与 `:563`），中间隔着一次完整的 chunk prefill，
  不可能合并成一次 launch；数据流方向也相反（池→计算缓冲 / 计算缓冲→池）。
- 两者可以共用页寻址与 stride 处理代码，但接口、`validate_data` 语义、replay 语义相互独立，
  需要分别登记、分别验收。
- 请不要只实现其中一个：只做 scatter 仍会留下 30 ms/层的 gather（见「性能基线」）。

## 背景与功能说明
使用场景：GLM-5.3-Flash 的 KDA 线性注意力层每步都要把「序列的递归状态页」在状态池与计算缓冲之间搬运一次
（prefill 每层 1 次 gather + 1 次 scatter，decode 侧由 #04 覆盖）。TP8 下共 34 层。

当前问题：递归状态页位于分页状态池中，**页与页之间存在间隔**，因此 `state` 的 dim0 stride 不等于
`H*Dv*Dk`。通用索引算子在这种 stride 下无法按页做连续块搬运，实测单次调用 30–173 ms
（见「精度要求」下方的性能基线）。

目标功能：给定状态池 `state`、页索引 `indices` 和 `has_initial_state`，按 `indices` 逐页搬运，
一次调用完成全部序列的 gather / scatter，**只搬被选中的页**，且不要求 `state` 连续。

非目标：
- 不做状态更新计算（递推本身由 #03/#04 负责）；
- 不支持 dtype 转换（纯搬运，不改变数值）；
- 不负责构造 `indices`（页索引由调用方给出）。

## 期望 API 与函数签名
```python
def dlc_kda_state_gather(
    state: torch.Tensor,              # [num_pages, H, Dv, Dk] fp32
    indices: torch.Tensor,            # [num_seqs] int32 / int64
    has_initial_state: torch.Tensor,  # [num_seqs] bool（也可接受 int32）
    out: torch.Tensor | None = None,  # 可选，[num_seqs, H, Dv, Dk] fp32
) -> torch.Tensor:                    # [num_seqs, H, Dv, Dk] fp32

def dlc_kda_state_scatter(
    state: torch.Tensor,              # [num_pages, H, Dv, Dk] fp32，原位写回
    src: torch.Tensor,                # [num_seqs, H, Dv, Dk] fp32
    indices: torch.Tensor,            # [num_seqs] int32 / int64
    validate_data: bool = False,
) -> None:
```

## 参数与返回值
输入：
- `state`: torch.Tensor，状态池；dim0 为页号，页内为 `[H, Dv, Dk]`；`stride(0)` 可能大于 `H*Dv*Dk`
- `indices`: torch.Tensor，每序列使用的页号；允许出现 `-1`（`PAD_SLOT_ID`，表示该序列本步无可用页）
- `has_initial_state`: torch.Tensor，逐序列标记是否已有历史状态；为假时 gather 输出该序列全 0
- `src`: torch.Tensor（仅 scatter），逐序列要写回的页内容

属性：
- `out`: torch.Tensor 或 None，gather 的可选输出缓冲；为 None 时算子自行分配
- `validate_data`: bool，默认 `False`；开启时做设备侧索引范围与重复页校验

返回值：
- gather: `out`，`[num_seqs, H, Dv, Dk]` fp32
- scatter: 无返回值，原位写回 `state`

## 功能语义
```text
gather : out[s] = (indices[s] >= 0 and has_initial_state[s]) ? state[indices[s]] : 0
scatter: indices[s] < 0 → 跳过该序列（state 不变）
         否则             state[indices[s]] = src[s]
```

- 寻址规则：`state` 的第 `p` 页起始元素 = `p * state.stride(0)`，页内按 `[H, Dv, Dk]` 连续读取 / 写入；
  **禁止**把 `state` 整池搬一遍（只搬 `num_seqs` 页）。
- in-place / alias：gather 不改写 `state`；scatter 原位改写 `state`，`src` 只读；
  允许 `src` 与 `out`/`state` 不重叠；不支持 `src is state` 的别名。
- mask / 广播：`has_initial_state` 为假时输出整页按 0 填写（不做 broadcast）。
- 空张量：`num_seqs == 0` 时直接返回（scatter 为 no-op）；`num_pages` 可为 0，此时任何非负 `indices` 都是越界。
- `indices` 含重复页：gather 允许（各自独立读取）；scatter 语义未定义，
  `validate_data=True` 时应报错。
- 异常输入：越界 `indices` 在 `validate_data=True` 时报错；`validate_data=False` 时行为未定义但不允许越界写。
- 确定性与随机数：纯数据搬运，无随机数，结果与执行顺序无关。

## 支持范围与约束
各参数支持的 dtype：
- `state`、`src`、`out`：fp32（递归状态需保精度），要求 bit-exact 搬运；不接受 bf16/fp16 输入
- `indices`：int32 或 int64
- `has_initial_state`：bool 或 int32

各参数支持的 shape 及动态范围：
- `state`：`[num_pages, H, Dv, Dk]`，`num_pages` 动态（真机 1168；须支持更大页数，不假设上界）
- `indices` / `has_initial_state`：`[num_seqs]`，`num_seqs` 动态（真机单请求为 1，需支持批量）
- `src` / `out`：`[num_seqs, H, Dv, Dk]`，必须与 `state` 的页内形状一致
- 真机固定值：`H = 8`（TP8，每卡 8 个头）、`Dv = Dk = 128`；页内单页 512 KiB
- 要求 `H*Dv*Dk` 可配置，不能硬编码为定值

数值范围：与输入一致，无中间计算，无溢出风险。

layout、stride、contiguous 和 device 要求：
- **`state` 必须支持 dim0 非连续**：按 `state.stride(0)` 定位页，不能假设等于 `H*Dv*Dk`；
  真机 `state` 是分页缓冲（页大小 542720 B = conv 段 18432 B + rec 段 524288 B）内的视图，
  等效 `stride(0) = 135680`（元素），而页内 `8*128*128 = 131072`；
- 页内 `[H, Dv, Dk]` 三维护视为连续，可整体做一次 512 KiB 连续块搬运；
- `indices` / `has_initial_state` 必须为连续张量；
- 所有输入输出同 device（DLC 单卡内）。

单卡/多卡及芯片型号：单卡调用，图内按 rank 独立执行（TP8 共 8 张卡）。

其他限制：
- **算子内部不得出现 host D2H**（不允许 `.item()` / `.cpu()` / `torch.equal` 之类的真值判断）；
- 需可被 `torch.dlc.graph` 捕获并 replay，且 replay 前改写 `indices` 缓冲后结果正确；
- 越界 / 重复页校验只能放在 `validate_data=True` 分支内。

## 精度要求
### Golden reference（纯 PyTorch，CPU 可直接运行，无 DLC 依赖）

```python
import torch

PAD_SLOT_ID = -1

def kda_state_gather_ref(state: torch.Tensor, indices: torch.Tensor,
                         has_initial_state: torch.Tensor,
                         out: torch.Tensor | None = None) -> torch.Tensor:
    """state [P,H,Dv,Dk] (any stride(0)) -> out [S,H,Dv,Dk]"""
    P = state.shape[0]
    S = indices.numel()
    shape = (S,) + tuple(state.shape[1:])
    if out is None:
        out = torch.zeros(shape, dtype=state.dtype, device=state.device)
    else:
        out.zero_()
    for s in range(S):
        p = int(indices[s])
        if p == PAD_SLOT_ID:
            continue
        if not (0 <= p < P):
            raise IndexError(f"gather page index out of range: {p}")
        if bool(has_initial_state[s]):
            out[s].copy_(state[p])
    return out

def kda_state_scatter_ref(state: torch.Tensor, src: torch.Tensor,
                          indices: torch.Tensor) -> None:
    """state [P,H,Dv,Dk] in-place (any stride(0)); indices<0 -> skip"""
    P = state.shape[0]
    if src.shape[0] != indices.numel() or tuple(src.shape[1:]) != tuple(state.shape[1:]):
        raise ValueError("scatter shape mismatch")
    seen = set()
    for s in range(indices.numel()):
        p = int(indices[s])
        if p == PAD_SLOT_ID:
            continue
        if not (0 <= p < P):
            raise IndexError(f"scatter page index out of range: {p}")
        if p in seen:
            raise ValueError("duplicate writable KDA cache pages")
        seen.add(p)
        state[p].copy_(src[s])

if __name__ == "__main__":
    torch.manual_seed(0)
    P, H, Dv, Dk = 7, 2, 4, 5
    page = H * Dv * Dk
    # padded pool: page stride = page + 3 elements
    flat = torch.arange(P * (page + 3), dtype=torch.float32)
    state = flat[3:].view(P, H, Dv, Dk) if False else flat.as_strided((P, H, Dv, Dk), (page + 3, Dv * Dk, Dk, 1))
    idx = torch.tensor([3, PAD_SLOT_ID, 0, 5, 5])
    has = torch.tensor([True, True, False, True, True])
    out = kda_state_gather_ref(state, idx, has)
    assert torch.equal(out[1], torch.zeros((H, Dv, Dk)))          # pad slot -> zero
    assert torch.equal(out[2], torch.zeros((H, Dv, Dk)))          # no initial state -> zero
    assert torch.equal(out[0], state[3]) and torch.equal(out[4], state[5])
    src = torch.full((idx.numel(), H, Dv, Dk), 9.0)
    before = state.clone()
    sidx = torch.tensor([3, PAD_SLOT_ID, 0, 5, 6])          # scatter 不允许重复页
    kda_state_scatter_ref(state, src, sidx)
    assert torch.equal(state[3], src[0]) and torch.equal(state[0], src[2])
    assert torch.equal(state[5], src[3]) and torch.equal(state[6], src[4])
    assert torch.equal(state[1], before[1]) and torch.equal(state[4], before[4])   # untouched pages
    try:
        kda_state_scatter_ref(state, src, idx)              # idx 含重复页 5,5
        raise AssertionError("duplicate scatter page must raise")
    except ValueError:
        pass
    print("reference self-test OK  (strided state, stride0=%d, page=%d)" % (state.stride(0), page))
```

自检：把上面的代码存成 `ref_kda_state_gather_scatter.py` 后运行，或用同内容的仓库副本
`docs/glm5_flash/op_requests/refs/kda_state_gather_scatter_ref.py`：

```bash
python refs/kda_state_gather_scatter_ref.py
# reference self-test OK  (strided state, stride0=43, page=40)
```

### 对照：当前适配层实现（行为基准，逐字引用 `vllm_cl/patch/patch_kda.py:330-341`）

```python
def gather_initial_states_dlc(state, indices, has_initial_state):
    if indices.ndim != 1 or has_initial_state.shape != indices.shape: raise ValueError("native gather requires 1-D indices and mask")
    if torch.any(indices<0) or torch.any(indices>=state.shape[0]): raise IndexError("KDA gather page index out of range")
    result=state.index_select(0,indices.long()).clone(); mask=has_initial_state.bool().view(-1,*([1]*(state.ndim-1)))
    out=torch.where(mask,result,torch.zeros_like(result))
    return out

def scatter_states_dlc(state, src, indices):
    if indices.ndim != 1 or src.shape[0] != indices.numel() or src.shape[1:] != state.shape[1:]: raise ValueError("native scatter shape mismatch")
    if torch.any(indices<0) or torch.any(indices>=state.shape[0]): raise IndexError("KDA scatter page index out of range")
    if indices.unique().numel()!=indices.numel(): raise ValueError("duplicate writable KDA cache pages")
    state.index_copy_(0,indices.long(),src.to(state.dtype))
```

注：现状实现对 `indices<0` 是**直接报错**；本需求的语义按上面的 golden reference —— gather 时 `-1` 与
`has_initial_state=False` 都输出 0，scatter 时 `-1` 跳过。这是本算子相对现状的语义补充，请按 golden reference 实现。

输入数据分布、随机种子和样本数量：
- `state` 用 `torch.randn` 生成并固定 `torch.manual_seed`；
- `indices` 覆盖：全正、含 `-1`、乱序、重复（仅 gather）；
- `has_initial_state` 覆盖：全 True、全 False、混合；
- 样本数 ≥ 8 组，每组重复 5 次。

比较 dtype：fp32 输入 / fp32 输出，逐元素比较。

atol：0（纯搬运）
rtol：0（纯搬运）

必须逐元素一致的场景：全部场景，`state` 与 `out` 都要求与 reference **bit-exact**；
`has_initial_state=False` 的位置必须严格为 0。

NaN/Inf/极值/零值处理：原样搬运，不做任何数值变换；输入含 NaN/Inf 时输出对应位置同样为 NaN/Inf。

累加与中间计算精度：无算术运算。scatter 未选中的页必须保持原值不变（不得整池重写）。

性能基线（2026-09-29，TP0，prefill 单步含 34 层 gather+scatter，torch profiler 窗口内）：
| 调用 | 次数 | 合计 | 单次 |
|---|---|---|---|
| `gather`（当前 `index_select`+`where`） | 34 | 1.020 s | 30 ms |
| `scatter`（当前 `index_copy_`） | 34 | 5.889 s | 173 ms |
| 合计 | 68 | 6.909 s（占该 prefill 步 58%） | — |

同几何下「连续池 vs 页内 view」的对照（`probes/probe_state_pool_index_cost.py`，只搬 8 页）：
`index_select+clone` 0.181 ms → 579.7 ms；`index_copy_` 1.544 ms → 1873.0 ms。
本算子即要求在该几何下回到同一量级。

## 测试范围
- [x] 基本功能与 reference 对齐
- [x] 多 dtype 精度（fp32 输入 / int32+int64 索引 / bool+int32 mask）
- [x] 静态 shape 与动态 shape（`num_pages`、`num_seqs` 变化）
- [x] 最小值、最大值、空 tensor 等边界（`num_seqs=0`、`indices` 全 `-1`、`num_pages=0`）
- [x] 非 contiguous、特殊 stride 或 layout（`state.stride(0) != H*Dv*Dk`）
- [x] 非法参数和异常处理（越界、scatter 重复页，`validate_data=True`）
- [x] 单卡
- [ ] 多卡/分布式（按 rank 独立调用，无跨卡通信，可不覆盖）
- [x] 长时间稳定性
- [x] 性能基准与性能回归

具体测试用例、shape、数据范围和预期结果：
1. `state=[1168, 8, 128, 128] fp32`（连续）与同形状但 `stride(0)=135680`（页内 view），
   `num_seqs ∈ {1, 8}`，`indices ∈ {正序, 乱序, 含 -1}`，`has_initial_state ∈ {全 True, 全 False, 混合}`
   ⇒ 与 reference bit-exact。
2. `num_seqs=0` ⇒ gather 返回空张量，scatter 为 no-op。
3. `validate_data=True` + `indices >= num_pages` ⇒ 抛错；scatter 传入重复页 ⇒ 抛错。
4. 性能：真实几何下单次 gather ≤ 0.5 ms、scatter ≤ 1 ms；34 层累计 ≤ 50 ms/步
   （当前 6.909 s/步）。

## 验证环境与依赖版本
```text
DLC 硬件：8 卡（TP=8，无 PP）
vLLM commit：f5361fb09f
vLLM DLC（vllm-cl）commit：8b98f6d
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：VLLM_USE_DLC_COL_MAJOR_MATMUL=1
模型：GLM-5.3-Flash-Mix（linear_num_heads=64、linear_head_dim=128、TP8 → 每卡 H=8；num_blocks=1168）
```

## 验收标准
- [ ] API、参数校验和错误行为符合约定（含 `validate_data` 开关语义）
- [ ] 连续池与「页内 view（非连续 dim0）」两种几何下均与 reference bit-exact
- [ ] 真实几何下单次 gather ≤ 0.5 ms、scatter ≤ 1 ms；34 层累计 ≤ 50 ms/步
- [ ] 只搬被选中的页：scatter 未选中页保持原值，实测耗时与页数成正比、与池大小无关
- [ ] `validate_data=False` 时无 host D2H，可被 `DLCGraph` 捕获与 replay
- [ ] 文档、测试和示例已提交

## Reference、日志与附件
Reference 实现（native，行为基准）：
- gather：`vllm_cl/patch/patch_kda.py:330` `gather_initial_states_dlc`
- scatter：`vllm_cl/patch/patch_kda.py:337` `scatter_states_dlc`

复现 / 计时脚本：`docs/glm5_flash/op_requests/probes/probe_state_pool_index_cost.py`

```bash
cd /root/workspace/vllm-cl
python -u docs/glm5_flash/op_requests/probes/probe_state_pool_index_cost.py dlc:0
# 退出码 0 = 复现（padded 至少比 dense 慢 20x）
```

当前脚本输出（2026-09-28，dlc:0）：

```text
device=dlc:0  P=1168  state=[1168,8,128,128] fp32 (584 MiB)
  dense  stride=(131072, 16384, 128, 1)  contiguous=True
  padded stride=(135680, 16384, 128, 1)  contiguous=False

           index_select+clone    index_copy_
   dense             0.181ms        1.544ms
  padded           579.739ms     1873.024ms
   dense scatter 语义: OK
  padded scatter 语义: OK

放大倍数：index_select 3196x，index_copy_ 1213x
按 34 个 KDA 层累计（每层 1 次 gather + 1 次 scatter）：
  dense  单步     58.7 ms
  padded 单步  83393.9 ms
```

相关 kernel 源码（现状，供参考）：`DLC_Custom_Kernel/dlc_kernels/index/index_copy.c`
（`index_copy_vmask` 按整个 self 长度发起 DMA）、`index_select.c`。

真机 profile：`docs/glm5_flash/给算子组_05_state_pool_index_perf.md`。
