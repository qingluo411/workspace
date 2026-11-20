# fused_eh_norm_bf16 算子合并与验收记录

日期：2026-11-18  
范围：`DLC_Custom_Kernel`、`pytorch`、`vllm-cl`、GLM-5.3-Flash MTP  
目标算子：`torch.dlc_fused_eh_norm_bf16` / `custom_fused_eh_norm_bf16`

## 1. 结论摘要

- kernel 与 pytorch 的 EH norm 功能提交已合入并重新构建安装。
- 单算子静态 6/6、动态 4000/4000、issue 自包含脚本 90 组功能用例、
  200 组随机、12 组非连续 layout、6 类异常和 1000 次 Graph replay 全部通过。
- 性能达到 issue 目标：`T=4,H=4096` 为 **0.0207 ms**，相对 Python fallback
  **22.45x**；`T=40` 为 0.0200 ms，相对 fallback **23.45x**，无回退。
- vllm-cl 已把 `fused_eh_norm_dlc` 接到原生 op，DLC 测试 2/2 通过，探针确认
  实际调用原生 op。
- MTP 端到端使用固定 128-token prompt：eager **50/50**、FULL_DECODE_ONLY graph
  **20/20**、17K/34K/64K marker 全部命中。
- 发现一个与 EH norm 无关的前置缺口：当前 pytorch main 缺少 KDA chunk prefill
  14-arg checkpoint API，导致 kernel 参数数量不匹配并在
  `custom_dlc_kda_chunk_prefill` 卡死。补回 `ec181b5ff5` 后 KDA 静态 20/20 恢复。
- 8/64-token 短 prompt 的 MTP 输出仍与 no-MTP baseline 分叉；8-token 在强制
  Python EH norm fallback 时也分叉，说明该问题不由本算子引入，属于既有小 T
  MTP/KDA 路径，需另行排查。

## 2. 应用提交与构建

| 仓库 | 当前分支 | 提交 | 来源 | 说明 |
|---|---|---|---|---|
| `DLC_Custom_Kernel` | `release-glm53flash-3` | `9acc16cd2` | `b4376b161` | 新增 `fused_eh_norm_bf16.c`、dispatch 和测试 |
| `pytorch` | `main` | `5ce9732cb9` | `fcf197001a` | 新增 `torch.dlc_fused_eh_norm_bf16` aten op |
| `pytorch` | `main` | `51b4648508` | `ec181b5ff5` | 前置修复：KDA chunk prefill 14-arg checkpoint API |

构建结果：

- `DLC_Custom_Kernel/build.sh -j 8` 通过，`libcustom_dlc_perf_lib.so` 安装到
  `/usr/local/chipltech/synapse/lib`；`enabled_kernels.hpp` 包含
  `DispatchType::DLC` 的 `custom_fused_eh_norm_bf16`。
- `pytorch setup.py develop` 通过；最终 `torch.__version__ == 2.5.0`。
- `torch.dlc_fused_eh_norm_bf16` 与 `torch.ops.aten.dlc_fused_eh_norm_bf16`
  均可用。

## 3. 单算子验收

### 3.1 仓库自带 `pytorch_test`

| 测试 | 结果 |
|---|---:|
| `test_fused_eh_norm_bf16 --static --device 0` | **6/6 PASS** |
| `test_fused_eh_norm_bf16 --dynamic -i 200 -b 20 --device 0` | **4000/4000 PASS** |
| 动态覆盖率 | **91.56%** |

### 3.2 issue 自包含验证

脚本：`work-tests/glm53-flash/verify_fused_eh_norm_bf16.py`

| 项目 | 覆盖 | 结果 |
|---|---|---|
| 功能 | `T ∈ {0,1,2,4,8,16,32,40,64,128}` × `H ∈ {1024,2048,4096}` × 全 0/全非 0/混合 position | 90 组 PASS |
| 随机 | 200 组固定 seed，`T=40,H=4096`，混合 position | PASS |
| 精度 | BF16 vs FP32 reference，`atol=rtol=2e-2`；position 0 前半严格为 0 | PASS |
| Layout | `T=4/40`，`H=1024/4096`，row stride `H/H+1/2H` | 12 组 PASS |
| 异常 | positions 长度、shape、weight 长度、最后一维 stride、dtype、非有限 eps | 6/6 均抛 `RuntimeError` |
| Graph | `T=40,H=4096`，capture 后 1000 次 replay | PASS，地址稳定、输出一致 |

说明：`H+1` row stride 的正确性通过 `KernelDesc` 的设备侧连续化拷贝实现，不是
零拷贝路径；`Normalization.cpp` 中“原样传 descriptor”的旧注释已修正为实际语义。

### 3.3 性能

环境：DLC 单卡，`H=4096`，warmup 20、repeat 100，端到端同步计时。

| T | 原生 op (ms) | Python fallback (ms) | 加速比 |
|---:|---:|---:|---:|
| 4 | **0.0207** | 0.4656 | **22.45x** |
| 8 | **0.0182** | 0.4732 | **25.99x** |
| 16 | **0.0186** | 0.4715 | **25.29x** |
| 40 | **0.0200** | 0.4698 | **23.45x** |

issue 目标 `T=4 ≤ 0.2 ms`、相对 fallback `≥3x` 均满足；`T=40` 无回退。

## 4. vllm-cl 接入

`vllm_cl/patch/models/patch_glm5_mtp.py::fused_eh_norm_dlc` 现在在以下条件满足时
优先调用原生 op：

- 设备为 `dlc`；
- `inputs_embeds`、`previous_hidden`、两路 weight 为 BF16；
- `positions` 为 INT64；
- `H % 256 == 0`。

其他情况保留原 Python fallback，以兼容 CPU 单测和小 hidden 的调试用例。另加
`VLLM_DLC_NATIVE_EH_NORM=0` 仅用于 A/B 强制回退，默认走原生 op。

验证：

| 测试 | 结果 |
|---|---:|
| `pytest tests/test_glm5_mtp_fused_eh_norm.py` | **2/2 PASS** |
| DLC native-call 探针（`T=4,H=4096`） | `native_calls=1`，输出 `[4,8192]` |

## 5. MTP 端到端

配置：TP8、`max-model-len=66560`（长上下文）/ 4096（短请求）、
`max-num-seqs=1`、`max-num-batched-tokens=4096`（graph）/ 1024（eager）、
MTP `num_speculative_tokens=1`、prefix cache 关闭、`mamba-cache-mode none`。

固定 prompt：128 token，token IDs 为 `[100,101,...,227]`，greedy 32 token。
no-MTP eager baseline 输出为：

```text
126,228,126,229,126,230,126,231,126,232,126,233,126,234,126,235,...
```

结果：

| 场景 | 结果 | 备注 |
|---|---:|---|
| MTP eager，固定 128-token prompt，重复 50 次 | **50/50 PASS** | 单次 2.344–2.473 s |
| MTP + FULL_DECODE_ONLY graph，重复 20 次 | **20/20 PASS** | 单次 2.033–2.161 s |
| 长上下文 marker，17,025 token | **PASS** | 44.323 s，`NEEDLE-1000` |
| 长上下文 marker，34,025 token | **PASS** | 91.216 s，`NEEDLE-2000` |
| 长上下文 marker，64,625 token | **PASS** | 183.623 s，`NEEDLE-3800` |

## 6. 发现的问题

### 6.1 前置缺口：pytorch KDA checkpoint API 缺失

首次启动 MTP 服务时，首个请求在 `custom_dlc_kda_chunk_prefill` 卡死。直接跑
`test_kda_chunk_prefill --static` 报：

```text
aten::dlc_kda_chunk_prefill() expected at most 12 argument(s) but received 14 argument(s)
```

根因是 kernel 已合入 14-arg checkpoint ABI，而当前 pytorch main 缺少对应的
`ec181b5ff5`。补回并重建 pytorch 后：

- KDA schema 变为 14 参数（含两个 optional checkpoint 参数）；
- `test_kda_chunk_prefill --static` 恢复 **20/20 PASS**；
- MTP 服务不再卡在 KDA chunk prefill。

该修复是后续端到端验证的前置条件，不属于 EH norm 功能本身。

### 6.2 短 prompt 的 MTP 分叉

当前实测：

| prompt tokens | no-MTP baseline | MTP native | 是否一致 |
|---:|---|---|---|
| 8 | `198,154842,198,154842,...` | `198,197,197,18,...` | 否，首个分叉 token 1 |
| 64 | `67402,159,101,123,...` | `67402,159,101,102,...` | 否，首个分叉 token 3 |
| 128 | 见上文 | 与 baseline 完全一致 | 是，50/50 |

8-token prompt 在 `VLLM_DLC_NATIVE_EH_NORM=0`（强制 Python fallback）下仍与
baseline 分叉，且输出与 native 相同。因此该分叉不由 `fused_eh_norm` 引入，
更可能位于小 T 的 MTP target/KDA 路径，建议另开问题排查。

## 7. 证据文件

结果文件：

- `work-tests/glm53-flash/results/fused_eh_norm_bf16_20261118.json`
- `work-tests/mtp/results/fused_eh_norm_baseline_8_20261118.json`
- `work-tests/mtp/results/fused_eh_norm_baseline_64_20261118.json`
- `work-tests/mtp/results/fused_eh_norm_baseline_128_20261118.json`
- `work-tests/mtp/results/fused_eh_norm_mtp_eager_128_50_20261118.json`
- `work-tests/mtp/results/fused_eh_norm_mtp_graph_128_20_20261118.json`
- `work-tests/mtp/results/fused_eh_norm_mtp_graph_longctx_20261118.json`
- `work-tests/mtp/results/fused_eh_norm_mtp_eager_fallback_10_20261118.json`

关键日志：

- `/root/workspace/log/fused_eh_norm_nomtp_eager_20261118.log`
- `/root/workspace/log/fused_eh_norm_mtp_eager_native2_20261118.log`
- `/root/workspace/log/fused_eh_norm_mtp_graph66560_20261118.log`
- `/root/workspace/log/fused_eh_norm_mtp_eager_fallback_20261118.log`

## 8. 结论与后续

`fused_eh_norm_bf16` 的单算子功能、异常、layout、Graph 和性能均达到 issue 要求；
MTP eager 50/50、graph 20/20、长上下文 marker 在 128-token 固定 prompt 上通过。
建议关闭 EH norm 算子的验收项，同时保留以下两个独立事项：

1. 小 T（8/64）MTP 与 no-MTP baseline 分叉，需排查 MTP target/KDA 小 T 路径；
2. `vllm-cl` 的接入改动尚未提交（kernel/pytorch 功能提交已在本地分支）。
