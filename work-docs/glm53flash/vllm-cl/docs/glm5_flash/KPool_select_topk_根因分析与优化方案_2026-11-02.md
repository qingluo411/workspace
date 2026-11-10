# KPool `dlc_kpool_select_topk` 长上下文根因分析与优化方案

日期：2026-11-02  
范围：`DLC_Custom_Kernel/dlc_kernels/vllm/vllm_kpool_select_topk.cpp`  
目标：解释该算子为什么随上下文增长显著变慢，并给出分阶段优化与验收方案。

## 1. 结论

`dlc_kpool_select_topk` 的长上下文退化不是单点缺陷，而是以下成本叠加：

1. 每个 query token 扫描全部可见 pool，单 token 成本随 `context` 线性增长。
2. 只需要 top-512 个 pool，却对全部可见 pool 做完整稳定排序；实测换用 generic `bitonic_topk` 不降反升，说明该实现的排序常数也很高。
3. 外层按 token 循环，每个 token 重新读取全部可见 pool block，没有跨 token 复用。
4. pool/head 维串行执行，双 XYS 只切 token；decode `T=1` 时只有一个核工作。
5. 每个 pool 都有 block table 读取、`Assert`、取模、score 打包和权重读取等微开销。

近似成本：

```text
visible = seq_len / pool_size          # pool_size=4
单 token 成本 ≈ O(visible × H × D + visible × log²visible + out_width)
prefill 总成本 ≈ Σ visible(token) ≈ O(T²)
decode 总成本 ≈ O(batch × context)
```

## 2. 代码证据

主文件：`dlc_kernels/vllm/vllm_kpool_select_topk.cpp`

| 位置 | 代码行为 | 影响 |
|---|---|---|
| `:89` | 外层按 token 循环 | 每个 token 独立重扫 pool |
| `:111` | `visible = seql / pool_size` | 可见池数随上下文线性增长 |
| `:145-185` | `for p < visible`，内部 `for h < H` | pool/head 均为串行，只有 D=128 维向量化 |
| `:161-178` | 每个 head 单独点积、归约、ReLU、加权 | H=32 次串行点积/归约，缺少 head/pool tile |
| `:187-190` | `sort_line_i_stable(..., visible, false)` | 为 top-512 对全部可见池做完整稳定排序 |
| `:53-55` | `sort_len = max(1024, nearx2(max_visible))` | 32K 上下文时排序长度为 8192 |
| `:78-81` | 仅按 T 在双 XYS 间切分 | decode T=1 时 XYS1 空转 |
| `:86-87`、`:153-158` | 只缓存“最后一个 pool block” | 下一个 token 从 p=0 重新开始，跨 token 复用基本无效 |
| `:145-148` | 每个 p 读 `smem_table` 并做 `Assert` | 每 pool 的固定控制开销 |
| `:151-157` | 整块 `hbm_to_vmem` + `bf16ToF32` | 每个 token 重复搬运/转换可见 pool cache |

排序实现：`dlc_kernels/sort/sort.hpp:640-667` 使用稳定 bitonic 排序，`l2=max(1024,nearx2(len))`。长上下文时排序长度随可见池增长，而 top-k 只需要 512 个候选。

## 3. 量化

### 3.1 计算效率

32K context、32 请求 decode、单层：

- 可见池：`32768 / 4 = 8192`
- 点积工作量：`32 × 32 × 8192 × 128 ≈ 2.15 GFLOP`
- 当前 fast：约 `650 ms/层`，等效约 `3.3 GFLOPS`
- native：约 `92 ms/层`，等效约 `23 GFLOPS`
- 同环境 kernel log 的 matmul 可达 `500–600 GFLOPS`

4K prefill、单层：

- 4096 token，平均可见池约 512
- 点积工作量约 `17.2 GFLOP`
- 当前 fast：约 `7.745 s/层`，等效约 `2.2 GFLOPS`
- native：约 `0.435 s/层`，等效约 `39.5 GFLOPS`

结论：点积本身不是唯一瓶颈，串行扫描、完整排序和重复搬运把有效吞吐拉低了一个到两个数量级。

### 3.2 排序次数

- 4K prefill：约 `4096 token × 11 层 = 45,056` 次 1024 长排序。
- 32K context、32 请求 decode：`32 × 11 = 352` 次 8192 长排序。
- 实际只需要每行选出 512 个 pool。

### 3.3 数据搬运

`release-glm53flash` 的版本按单个 pool row 加载；当前 `main` 的 `0468479d0` 改为整 PPB block 加载，DMA 调用数约降低 `PPB=64` 倍。但两者都没有跨 token 复用，HBM 访问量仍随 `T × context` 增长。

若线上 `.so` 来自 release 分支，逐 row DMA 的调用次数会额外放大 64 倍，需要优先确认实际安装版本。

## 4. 根因排序

1. **token-major 重扫 + 串行 dot/reduction**：每个 token 重读全部可见 pool block，并对 H 个 head 逐个做 128 维归约；这是第一轮实测确认的主要成本。
2. **完整稳定排序**：仍然是长上下文的放大器；但 generic `bitonic_topk` 在 8192→512 形状上没有收益，不能只靠换排序库解决。
3. **pool/head 串行、双核利用不足**：最贵的 `visible` 维没有并行；decode T=1 时只有一个核工作。
4. **每 pool 控制开销**：table load、assert、取模、score 打包、权重读取。
5. **固定输出成本**：每个 token 清零/写回 2176 个 int，长上下文下放大但非首因。

## 5. 优化方案

### 阶段 1：已落地的低风险优化

先尝试了 `bitonic_topk` 替换 `sort_line_i_stable`。相同工具链、相同输入的 A/B 显示：

- 4K decode：`4.535 → 4.668 ms`，约慢 2.9%；
- 16K decode：`17.596 → 18.141 ms`，约慢 3.1%；
- 32 请求 32K decode：`558.485 → 576.263 ms`，约慢 3.2%；
- 4K prefill：`6.849 → 7.049 s`，约慢 2.9%。

原因是 `bitonic_topk` 在 8192→512 形状上仍要做输入拷贝、索引初始化、分段排序和多轮双调分裂，固定开销高于当前 full sort 的增量收益。因此已撤回该替换，避免把性能回退带进发布路径。

第一轮真正落地的是三处低风险改动：

1. **head 循环 8 路 unroll**：`#pragma clang loop unroll_count(8)`，让多个 head 的 128 维点积/归约可以重叠，降低串行依赖。
2. **把 `scale * head_weight_scale` 折进每 token 的 head 权重**：每 token 只缩放 H 个权重一次，去掉每个 pool×head 的两次标量乘法。
3. **block table/Assert 移到 PPB block 边界**：不再每个 pool 重复读取 `smem_table` 和做范围断言。

对应代码仍在 `dlc_kernels/vllm/vllm_kpool_select_topk.cpp`；没有改变 ABI、输出宽度、排序语义或 tie-break 规则。

`scale` 折叠依赖 `scale >= 0`（softmax scale 的约定），kernel 已加显式断言；`scale=0` 仍与 reference 等价，负 scale 不再支持。

### 阶段 2：query/pool tile 复用

- 外层改为 pool block 或 pool tile，内层处理多个 query token。
- 一次加载 pool block，复用于同一请求的多个 token；prefill 优先。
- decode 可将多个请求的 query 组成 tile，但需要按 block table 分组。
- DMA 与计算双缓冲，避免整块加载后串行等待。

### 阶段 3：提高并行度与矩阵效率

- decode 时按 pool 维切双 XYS，各自产生局部 top-k 后 merge。
- 复用 `kpool_mqa_logits_bf16.cpp` 的 RHST/tile 思路，把 H×P 点积交给矩阵单元。
- 减少每 pool 的标量归约和 score 打包，改成 tile 级累加。

### 阶段 2/3 原型结果（未合入）

已实现一个 `H=32、D=128、PPB=64` 的 RHST 原型：每 128 个 pool 一组，先把 1–2 个物理 block 装入 VMEM，再以 `head_tile=8` 调用 `matmul_input_bf16_out_f32_RHST`，最后沿用原稳定排序和 expand。原型在**单次 launch** 下通过了 syntests 9/9 和 pytorch 19/19，输出 hash 与 scalar 路径一致。

但连续 launch 会在第二次调用时触发设备 `trf0_of`（TRF overflow），说明当前把多组 matmul 直接塞进融合 kernel 的 tile/同步生命周期还不正确。该原型已回退，没有进入发布代码。后续若要继续矩阵化，必须满足：

- 将 score 计算拆成独立 kernel，或在每次 matmul 之间显式完成 TRF 排空；
- 用 standalone `kpool_mqa_logits` 已验证的 tile/同步路径，而不是在 `select_topk` 内重复调用；
- 单进程连续 launch 回归必须作为验收项，不能只测单次 launch。

## 6. 验收

正确性：

- `syntests -t vllm_kpool_select_topk` 全用例通过。
- `pytorch_test/test_kernel/vllm/test_vllm_kpool_select_topk.py` 全用例通过。
- 覆盖 ties、negative、空请求、非整池 top-k、`pool_size=1`、`D=64/128/256`、`H=2/8/16/32/64`。

性能：

- 4K/16K/32K prefill 单层耗时。
- 单请求与 32 请求、4K/16K/32K context decode 单层耗时。
- 记录 kernel cycle、DMA 次数和 sort/top-k 占比。

## 7. 第一轮实测

测试条件：同一台 8 卡机器、同一份新 LLVM 工具链、同一组确定性输入；旧版与新版的输出 SHA-256 完全一致。计时为单层 kernel wall time，含同步，不含模型其他算子。

| 场景 | 旧版 | 第一轮优化 | 变化 |
|---|---:|---:|---:|
| 单请求 4K decode | 4.535 ms | 4.213 ms | -7.1% |
| 单请求 16K decode | 17.596 ms | 16.296 ms | -7.4% |
| 32 请求 32K decode | 558.485 ms | 517.097 ms | -7.4% |
| 4K prefill（T=4096） | 6,849.3 ms | 6,350.5 ms | -7.3% |
| 32K 上下文 prefill（T=4096） | 69,220 ms | 64,038 ms | -7.5% |

正确性：

- `syntests -t vllm_kpool_select_topk`：9/9 通过，覆盖 ties、negative、空请求、非整池 top-k、`D=64/128/256`、`H=2/8/16/32/64`。
- `pytorch_test` static：19/19 通过，覆盖 `scale=0`、`head_weight_scale=0`、非整池 top-k 和不等长请求。
- 新旧输出逐字节 hash 全部一致。

结论：

- 第一轮优化是“降低常数项”，不是改变复杂度；对长上下文仍有约 7.5% 的稳定收益，值得保留。
- 要获得数量级提升，必须进入阶段 2/3：query/pool tile 复用、减少 token-major 重扫，以及使用 `kpool_mqa_logits_bf16.cpp` 的 RHST/tile 把 H×P 点积交给矩阵单元。
- 当前不应继续在 full sort 上做局部替换；generic top-k 已证明不是该形状的收益点。
- RHST 原型证明了 score 计算可以矩阵化，但连续 launch 的 TRF 生命周期尚未解决，因此未合入；下一步优先复用 standalone `kpool_mqa_logits` 或拆成独立 score/topk kernel。
