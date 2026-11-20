# KDA causal-conv checkpoint 算子合并与验收记录

日期：2026-11-18  
范围：`DLC_Custom_Kernel`、`pytorch`、`dlc_kda_causal_conv_prefill`  
目标：补齐 KDA prefill checkpoint issue 中的 conv state 导出能力

## 1. 结论摘要

- 远端 kernel/pytorch 目标提交已 cherry-pick 并重新构建安装。
- checkpoint 静态 **10/10**、动态 **4000/4000** 全过，覆盖率 **91.63%**。
- 非 checkpoint causal-conv、causal-conv update、recurrent decode 回归全部通过。
- 独立对拍确认 checkpoint conv state 与“只跑前缀的同一算子 final state”
  **bit-equal**，包括 4K/64K、真实 `D=3072`、batch mixed offset 和 scratch 行。
- `checkpoint_offsets=None` 与全 0 offsets 的输出/state 完全一致，checkpoint 不写。
- 额外 kernel launch 为 **0**；4K 和 64K（真实 `D=3072`）checkpoint 路径开销约
  **0%**，满足 issue 的 `<5%` 目标。
- 非法非 64 对齐 offset 会触发明确的 device assert，并返回 `RuntimeError`；
  C++ 参数组合、shape、dtype、非连续 checkpoint_state 均会在 launch 前报错。

## 2. 应用提交与构建

| 仓库 | 当前分支 | 本地提交 | 来源 | 说明 |
|---|---|---|---|---|
| `DLC_Custom_Kernel` | `release-glm53flash-3` | `4de635bbb` | `3ae8cbea7` | causal-conv checkpoint kernel + 测试 |
| `pytorch` | `release-glm53-flash` | `548bc4180c` | `9a5a5d3111` | causal-conv checkpoint API |

kernel 前置 `10b00d9f7` 与本地 `bb85a571f` 的 patch-id 完全一致，未重复合入。

构建：

- `DLC_Custom_Kernel/build.sh -j 8` 通过，安装
  `/usr/local/chipltech/synapse/lib/libcustom_dlc_perf_lib.so`。
- `pytorch setup.py develop` 通过，`torch.__version__ == 2.5.0`。
- `torch.ops.aten.dlc_kda_causal_conv_prefill` schema 已包含
  `checkpoint_offsets` / `checkpoint_state` 两个 optional 参数。
- `ninja -C build syntests -j 16` 构建通过。

## 3. 官方测试

### 3.1 `pytorch_test`

| 测试 | 结果 |
|---|---:|
| `test_kda_causal_conv_prefill_checkpoint --static` | **10/10 PASS** |
| `test_kda_causal_conv_prefill_checkpoint --dynamic -i 200 -b 20` | **4000/4000 PASS** |
| checkpoint 动态覆盖率 | **91.63%** |
| `test_kda_causal_conv_prefill --static`（无 checkpoint 回归） | **10/10 PASS** |
| `test_kda_causal_conv_update --static` | **16/16 PASS** |
| `test_dlc_kda_recurrent_decode --static` | **12/12 PASS** |

checkpoint 静态用例包含：

- 单请求 4K：`T=4096, D=3072, offset=3584`；
- 单请求 64K：`T=65536, D=256, offset=65280`；
- batch mixed offset：`lengths=[4096,7168], offsets=[3584,6912]`；
- `offset=0`、null slot、`state_len=80` 的 scratch 行保留；
- mid-tile checkpoint，验证不会覆盖已写出的卷积输出行。

### 3.2 C++ syntests

```text
./build/syntests/syntests -t kda_causal_conv_prefill
./build/syntests/syntests -t kda_causal_conv_update
```

| 测试 | 结果 |
|---|---:|
| `kda_causal_conv_prefill` | **12/12 cases PASS** |
| `kda_causal_conv_update` | **7/7 cases PASS** |

prefill syntest 的 `kda_prefill_out` 最大误差为 `7.22e-3`（阈值 `3e-2`）；
`kda_prefill_state` 和 `kda_prefill_checkpoint` 全部为 **0.0** 精确一致。

## 4. issue 自包含验证

脚本：`work-tests/glm53-flash/verify_kda_causal_conv_checkpoint.py`

### 4.1 checkpoint state vs prefix-only final state

| 场景 | checkpoint vs prefix final state |
|---|---:|
| `T=4096, D=3072, offset=3584` | max abs diff **0.0**，bit-equal |
| `T=65536, D=256, offset=65280` | max abs diff **0.0**，bit-equal |
| `T=65536, D=3072, offset=65280` | max abs diff **0.0**，bit-equal |
| batch `lengths=[4096,7168], D=512, offsets=[3584,6912]` | 两行均为 **0.0**，bit-equal |
| `state_len=80, lengths=[128,128], offsets=[64,64]` | 两行均为 **0.0**，bit-equal，scratch 行保留 |

说明：官方阈值允许 `2e-2`，本实现实测为 bit-equal，优于 issue 要求。

### 4.2 no-checkpoint / zero-offset 等价

`T=4096, D=3072` 下：

- `checkpoint_offsets=None` 与 `checkpoint_offsets=[0]` 的 `out` 完全相等；
- 两种路径的 conv `state` 完全相等；
- `checkpoint_state` 的初始哨兵值 `3.0` 保持不变，确认 offset 0 不写 checkpoint。

### 4.3 性能

同一 kernel launch，checkpoint 仅增加 kernel 内快照和 state 写回，不增加 launch。

| 场景 | 无 checkpoint | 开启 checkpoint | 额外开销 | 额外 launch |
|---|---:|---:|---:|---:|
| `T=4096, D=3072, offset=3584` | 2.7895 ms | 2.7840 ms | -0.20%（测量噪声） | 0 |
| `T=65536, D=256, offset=65280` | 19.3930 ms | 19.3929 ms | ≈0% | 0 |
| `T=65536, D=3072, offset=65280` | 43.9328 ms | 43.9348 ms | +0.005% | 0 |

满足 issue 的“额外 launch 不超过 1、额外 prefill 开销小于 5%”。

### 4.4 参数与错误行为

以下情况均返回 `RuntimeError`：

- 只传 `checkpoint_offsets`，不传 `checkpoint_state`；
- 只传 `checkpoint_state`，不传 `checkpoint_offsets`；
- `checkpoint_state` shape 错误；
- `checkpoint_state` dtype 错误；
- `checkpoint_state` 非 DLC contiguous；
- `checkpoint_offsets` dtype 错误（非 int32）。

另测 `T=64, offset=1`（非 64 对齐）：

```text
checkpoint offset must be 0 or a positive multiple of 64 below sequence length
DLC error: synErrorLaunchFailure
```

该路径由 kernel Assert 覆盖，负 offset 和 `offset >= sequence length` 使用同一断言。

## 5. 结论

`dlc_kda_causal_conv_prefill` 的 conv checkpoint 能力满足 issue 的功能、精度、
batch mixed offset、no-checkpoint 回归和性能要求。与 `dlc_kda_chunk_prefill`
的 recurrent checkpoint 组合后，KDA prefill checkpoint 的算子侧 conv + recurrent
API 已经闭环。

后续框架侧仍需：

1. 在 vLLM DLC 路径把 `checkpoint_offsets` / `checkpoint_state` 接到 causal conv
   原生算子，删除 Python split fallback；
2. 打开 MTP target 的 KDA checkpoint gate；
3. 重新验收 4K/64K 首次请求导出、第二次请求命中、冷/热 greedy 输出一致。

## 6. 证据文件

- `work-tests/glm53-flash/results/kda_causal_conv_checkpoint_20261118.json`
- `/tmp/kda_ckpt_verify.out`

关键代码：

- kernel：`dlc_kernels/conv/kda_causal_conv/kda_causal_conv_prefill.c`
- pytorch：`aten/src/ATen/native/dlc/CausalConv1d.cxx`
- 官方测试：`pytorch_test/test_kernel/conv/test_kda_causal_conv_prefill.py`
