# KDA prefill checkpoint 算子合并与验收记录

日期：2026-11-17  
范围：`DLC_Custom_Kernel`、`pytorch`、KDA prefill checkpoint issue

## 1. 合并内容

远端分支：

- kernel：`origin/baidiao_feat_8476_261116`
- pytorch：`origin/baidiao_feat_2269_261116`

两条远端分支都夹带了与 KDA checkpoint 无关的提交。本次只取 KDA 功能提交并
cherry-pick：

| 仓库 | 当前分支 | 本地提交 | 来源提交 |
|---|---|---|---|
| `DLC_Custom_Kernel` | `release-glm53flash-3` | `7894c5129` | `830835ae8` |
| `pytorch` | `main` | `ec181b5ff5` | `3446511336` |

`DLC_Custom_Kernel` 的 kernel 改动只涉及
`dlc_kernels/attention/kda/dlc_kda_chunk_prefill.cpp`，pytorch 改动只涉及
`aten/src/ATen/native/dlc/FusedOps.cpp` 和 `native_functions.yaml`。

## 2. 构建与基础测试

构建通过：

```text
DLC_Custom_Kernel/build.sh                    PASS
pytorch setup.py develop                      PASS
syntests target `syntests`                    PASS
```

测试结果：

| 测试 | 结果 |
|---|---:|
| `test_kda_chunk_prefill --static --device 0` | 20/20 PASS |
| `test_kda_chunk_prefill --dynamic --iterations 200 --batch-size 20 --device 0` | 4000/4000 PASS |
| syntest `dlc_kda_chunk_prefill` | 5/5 PASS |

动态测试覆盖了 batch 1/2、heads 1/2/4/8、chunk 16/32、不同 token 长度和
checkpoint offset；静态测试的 checkpoint state 对拍也通过。

## 3. 补充验证

### 3.1 真实 4K 规格

- `T=4096, H=8, D=128, chunk=64, offset=3584`
- 与 CPU prefix reference 对比：
  - `out` max abs error：`0.000488`
  - `final_state` max abs error：`0.002571`
  - `checkpoint_state` max abs error：`0.002510`（低于 issue 的 `2e-2`）

### 3.2 真实 heads=64

- `T=4096, H=64, D=128, chunk=64, offset=3584`
- `checkpoint_state` 与同一算子对 `[0,3584]` 前缀运行的 `final_state`：
  - max abs error：`0.0`
- 两者均为 finite。

### 3.3 64K 长序列 smoke

- `T=65536, H=1, D=128, chunk=64, offset=65280`
- `checkpoint_state` 与同一算子前缀运行的 `final_state`：
  - max abs error：`0.0`
- 结果 finite；该测试验证长序列 launch/索引，不做完整 64K CPU reference。

### 3.4 Batch 内混合 offset

- `B=2, T=256, offsets=[128, 0]`
- `checkpoint_state` 与 CPU prefix reference：
  - max abs error：`0.002847`
- `offset=0` 的 batch row 输出保持全 0。

### 3.5 可选参数兼容

- 不传 `checkpoint_offsets` / `checkpoint_state`：调用成功，输出和 final state 正常；
- 传 `checkpoint_offsets=[0,0]` 和全 0 `checkpoint_state`，与不传可选参数对比：
  - `out` max abs diff：`0.0`
  - `final_state` max abs diff：`0.0`
  - checkpoint 全 0。

## 4. 未满足项与风险

### 4.1 `dlc_kda_causal_conv_prefill` 没有 checkpoint 扩展

这是当前最主要的缺口。需求文档要求两个算子都支持
`checkpoint_offsets` / `checkpoint_state`，但合并后的 API 和 kernel 仍有：

```text
dlc_kda_causal_conv_prefill(qkv, weight, conv_state, query_start_loc,
                            state_indices, has_initial_state, ...)
```

没有 optional checkpoint 参数，也没有对应的 conv checkpoint state 输出逻辑。
pytorch checkpoint 测试只覆盖 `custom_dlc_kda_chunk_prefill`，syntest 也只改了
chunk prefill。因此当前只能把本轮工作判定为“chunk recurrent checkpoint 部分
满足，conv checkpoint 未交付”。

### 4.2 官方测试矩阵与 issue 要求不完全一致

issue 要求覆盖 `T=4096/65536`，但新增的 `test_kda_chunk_prefill.py` 静态参数最大
只到 257，动态参数也只到 68。4K/64K 的结果来自本次额外 probe，不是仓库内可
持续运行的 CI 测试。

### 4.3 API 安全边界需要算子侧确认

kernel 中 `has_checkpoint_state` 变量未使用，且 `checkpoint_offsets_hbm` 在
`checkpoint_offsets=None` 时仍会被传给 `load_i32_scalar_kda()`。本次 smoke 没有
观察到崩溃，数值也与无 checkpoint 路径一致，但建议算子侧：

- 只在 `checkpoint_offsets != null` 时读取 offset；
- 明确要求非零 offset 必须同时提供 `checkpoint_state`，或显式忽略；
- 对 offset 小于 0、超过 query 长度、非 64 对齐给出明确错误。

### 4.4 性能要求尚未正式量化

本次只做了非正式首轮计时，不能据此声称满足 issue 的“额外 launch 不超过 1、
额外 prefill 开销小于 5%”。chunk kernel 的 checkpoint 只是主 kernel 内部条件
写回，理论上不增加 launch，但仍需标准 warmup + 多次重复的 `B=1,T=65536` 数据。

### 4.5 现有 causal-conv 回归测试本身不干净

在本次 merge 后的环境执行：

```text
test_kda_causal_conv_prefill --static --device 6
```

结果为 `3 PASS / 7 FAIL`。该提交没有修改 causal-conv kernel 或其测试，失败更像是
既有测试/环境问题，不是本次 checkpoint 改动引入；但在把整个 issue 标记为完成
前需要单独确认基线并清理。

## 5. 结论

`dlc_kda_chunk_prefill` 的 checkpoint 输出已经可用：构建、静态 20/20、动态
4000/4000、syntest 5/5 通过，4K/64K 和 batch mixed-offset 补充验证也通过。
但需求文档要求的 `dlc_kda_causal_conv_prefill` checkpoint 仍未实现，官方测试也
缺 4096/65536 覆盖，性能指标尚未正式测量。因此本次合并**部分满足需求，不能
直接关闭完整 KDA prefill checkpoint issue**。
