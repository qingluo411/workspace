# GLM-5.3-Flash 异步 H2D 与 Graph 启动卡死问题

日期：2026-11-20  
现场采集时间：2026-11-20 13:00-13:03（Asia/Shanghai）  
状态：两组问题均已按复现命令验证；最后一次问题二验证实例当前仍处于卡住状态，
未停止进程、未再次 reset 设备。

本文记录两个需要分开处理的问题：

1. **MTP `num_speculative_tokens > 1` + `FULL_DECODE_ONLY` graph 时，draft metadata 的 `query_start_loc` 被错误消费。**
2. **MTP=1 + graph 启动阶段的概率性卡死，设备停在 `custom_gather_tensor` 的 `bad_vld_base: Vld base out of range`。**

两者都发生在 graph capture/warmup 和异步 metadata/copy 附近，但报错层级、触发条件和当前证据不同，不能直接视为同一个 root cause。

---

## 1. 当前运行配置

### 1.1 服务命令

```bash
vllm serve /data/models/GLM-5.3-Flash-Mix \
  --served-model-name GLM-5.3-Flash-Mix \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 128000 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --port 18005 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kernel-config '{"enable_jit_warmup": false}'
```

### 1.2 关键环境变量

API 进程显式设置了：

```text
DLC_SYN_COPY_ASYNC=O3
DLC_SYN_GRAPH_CACHE=1
DLC_MTP_DRAFTER_CUDAGRAPH=1
DLC_MTP_PROPOSER_CUDAGRAPH=1
DLC_KPOOL_SLOT_MAPPING_CHECK=1
VLLM_USE_DLC_COL_MAJOR_MATMUL=1
PYTORCH_DLC_ALLOC_CONF=expandable_segments:True
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
```

从 `/proc/<pid>/environ` 的初始环境看，EngineCore 和 TP worker 没有显式列出
`DLC_SYN_COPY_ASYNC`。PyTorch 在变量缺失时默认也是 `O3`，因此当前现场大概率仍为
O3 语义；但后续做 O0/O2/O3 对照时必须确认该变量确实传到了真正执行 copy 的 worker
进程，不能只设置在 API 父进程。

### 1.3 代码与产物

| 组件 | 分支 | revision |
|---|---|---|
| vllm | `release-glm53flash` | `d32125aa2b` |
| vllm-cl | `release-glm53flash` | `adbb306` |
| DLC_Custom_Kernel | `release-glm53flash-3` | `a276dc22e` |
| pytorch | `release-glm53-flash` | `546c065429` |
| DLCSynapse | `main` | `7500cf64` |

已安装算子库：`/usr/local/chipltech/synapse/lib/libcustom_dlc_perf_lib.so`，
大小 `1,616,055,088` B，mtime `2026-11-20 03:36`。

### 1.4 首次现场进程状态

以下 PID 属于第一次卡住现场，后续为复现问题一已主动停止，不是当前运行实例。

| PID | 进程 | 当前状态 |
|---:|---|---|
| `854799` | `vllm serve` API | 在 `wait_for_engine_startup()` 等待 EngineCore |
| `855264` | `VLLM::EngineCore` | 在 `determine_available_memory()` 等待 worker RPC |
| `855709` | `VLLM::Worker_TP0` | 在 `_dummy_run()` 的 `torch.dlc.synchronize()` 等待 |
| `855710` | `VLLM::Worker_TP1` | Python 停在 GDN metadata 构建；native 栈在 H2D copy 同步 |
| `855711` | `VLLM::Worker_TP2` | 同 TP1 |
| `855712` | `VLLM::Worker_TP3` | 在 `_dummy_run()` 的 `torch.dlc.synchronize()` 等待 |
| `855713` | `VLLM::Worker_TP4` | 同 TP3 |
| `855714` | `VLLM::Worker_TP5` | 同 TP3 |
| `855715` | `VLLM::Worker_TP6` | 同 TP1 |
| `855716` | `VLLM::Worker_TP7` | 同 TP3 |

API 尚未完成 engine startup，日志中不可能出现 `Application startup complete`。当前服务
stdout/stderr 指向 `/dev/pts/4`，没有落盘日志文件；本次未读取该 pty，避免消费终端输出。

---

## 2. 问题一：`num_speculative_tokens > 1` + graph 的 qsl 错误

### 2.1 现象

`num_speculative_tokens=2` + `FULL_DECODE_ONLY` graph 在 graph capture 阶段确定复现：

```text
RuntimeError: kpool_slot_mapping: query_start_loc[0] must be 0, got 16
```

该错误在同步调度下也能复现，因此不是 async scheduling 独占的问题。

其中 `got` 后面的数值取决于当前 graph bucket/batch 配置。历史 `max-num-seqs=16`
时为 `16`；本文给定的 `max-num-seqs=8` 复现命令实际得到 `8`。错误点和错误条件不变。

### 2.2 已验证事实

1. graph capture 中同时存在 target/verify 和 draft 两组 metadata：
   - target/verify：`num_reqs=16, num_tokens=48`，qsl 期望为 `[0,3,6,...,48]`；
   - draft：`num_reqs=16, num_tokens=16`，CPU qsl 期望为 `[0,1,2,...,16]`。
2. draft 的 CPU qsl 始终正确，但 GPU qsl 曾被观察到变成 `[1,1,...,0]`、随机
   int32，或从错误偏移开始，使 kernel 看到的首元素是 `16`。
3. 在 `_dummy_run` 中重置 qsl、填充 padding 并执行 `copy_to_gpu()` 后，GPU qsl 仍可能
   被后续 metadata/copy 覆盖。
4. 设置 `DLC_SYN_COPY_ASYNC=O0` 后仍报同一错误，说明问题不只是 DLC copy 的
   O0/O3 执行级别。
5. 设置 `DLC_KPOOL_SLOT_MAPPING_CHECK=0` 可以让 capture 完成，但 `num_spec=2` 的
   输出从第 2 个 token 起与 no-MTP 不同；因此关校验只是隐藏错误，不是修复。

### 2.3 当前结论

错误发生在 KPool #19 的参数校验，但根因不在 KPool kernel。更符合证据的解释是：

> graph capture 期间，draft `query_start_loc` 使用的是临时/共享 metadata staging；
> capture/replay 生命周期内该 buffer 又被 target/draft 的后续 metadata 覆盖，导致
> kernel 实际消费到的 qsl 与 Python CPU 侧看到的不一致。

相关代码路径包括：

- `vllm/v1/spec_decode/llm_base_proposer.py:1249-1254`
  - `token_indices`、`query_start_loc`、`seq_lens` 通过 `async_tensor_h2d()` 发送到 device。
- `vllm/utils/torch_utils.py:711-731`
  - `async_tensor_h2d()` 默认执行 `t.to(device, non_blocking=True)`。
- `vllm-cl/vllm_cl/worker/dlc_model_runner.py:5829-5830`
  - `_dummy_run()` 更新共享的 `self.query_start_loc.np` 后执行 `copy_to_gpu()`。
- `pytorch/aten/src/ATen/native/dlc/FusedOps.cpp:7847`
  - KPool slot mapping 的 `query_start_loc[0] == 0` 检查。

### 2.4 复现命令

环境变量：

```bash
export VLLM_PLUGINS=dlc,dlc_patches
export DLC_SYN_GRAPH_CACHE=1
export DLC_SYN_COPY_ASYNC=O0
export DLC_MTP_DRAFTER_CUDAGRAPH=1
export DLC_MTP_PROPOSER_CUDAGRAPH=1
export VLLM_USE_DLC_COL_MAJOR_MATMUL=1
export DLC_KPOOL_SLOT_MAPPING_CHECK=1
export PYTORCH_DLC_ALLOC_CONF=expandable_segments:True
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_LOGGING_LEVEL=INFO
```

启动命令：

```bash
vllm serve /data/models/GLM-5.3-Flash-Mix \
  --served-model-name GLM-5.3-Flash-Mix \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 128000 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --port 18005 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kernel-config '{"enable_jit_warmup": false}'
```

预期在 graph capture 阶段出现：

```text
RuntimeError: kpool_slot_mapping: query_start_loc[0] must be 0, got 8
```

`got` 的数值随 batch/graph bucket 变化；在 `max-num-seqs=8` 的复现命令下为 `8`。

### 2.5 当前适用范围

- `num_speculative_tokens=1 + graph`：当前稳定可用，也是现有最佳性能配置采用的路径。
- `num_speculative_tokens=2 + graph`：已确定失败。
- `num_speculative_tokens=4/8`：只完成 eager 基础正确性，graph 仍不能宣称可用。

潜在修复方向是：在进入 graph capture 前，为每个 graph bucket 预置持久化的 qsl 和
seq-lens staging buffer，并保证整个 graph 生命周期内地址不被共享 metadata 覆盖；
capture 后只通过受控 copy 更新内容。单纯把 `async_tensor_h2d()` 改成一次普通同步 copy，
不足以解决 buffer 生命周期问题。

---

## 3. 问题二：MTP=1 + graph 启动概率性卡死和 `bad_vld_base`

### 3.1 现象

在当前 MTP=1、prefix cache、`FULL_DECODE_ONLY` graph 的启动过程中，服务停在 graph
memory profiling 的 warmup 阶段。本次已运行约 8 分钟仍未推进，属于确定性卡住现场。

芯片状态：

| Device | State | Errors | 当前 kernel / 状态 |
|---:|---|---|---|
| cltech0 | Running | PASS | `custom_allreduce_2x4tpu4root_simple_bf16_rdma` |
| cltech1 | Halted | **FAIL** | `custom_gather_tensor`, `bad_vld_base` |
| cltech2 | Halted | **FAIL** | `custom_gather_tensor`, `bad_vld_base` |
| cltech3 | Running | PASS | `custom_allreduce_2x4tpu4root_simple_bf16_rdma` |
| cltech4 | Running | PASS | `custom_allreduce_2x4tpu4root_simple_bf16_rdma` |
| cltech5 | Running | PASS | `custom_allreduce_2x4tpu4root_simple_bf16_rdma` |
| cltech6 | Halted | **FAIL** | `custom_gather_tensor`, `bad_vld_base` |
| cltech7 | Running | PASS | `custom_allreduce_2x4tpu4root_simple_bf16_rdma` |

1/2/6 的 XYS0/XYS1 均报告：

```text
bad_vld_base: Vld base out of range
```

### 3.2 卡住时的算子参数

cltech1、cltech2、cltech6 均停在 `custom_gather_tensor nbytes=488`。三张卡的主要输入形状一致：

```text
DLCTensor[1]:
  shape=[501, 8, 1, 1, 1]
  stride=[1, 501, 4008, 4008, 4008]
  dtype=DLCInt32

DLCTensor[1]:
  shape=[2, 4, 1, 1, 1]
  stride=[1, 2, 16, 16, 16]
  dtype=DLCInt64

DLCTensor[1]:
  shape=[2, 8, 1, 1, 1]
  stride=[1, 2, 16, 16, 16]
  dtype=DLCInt32
```

`501` 与当前配置的 KDA raw block table 宽度吻合：

```text
128000 / 256 + 1 = 501
```

结合已有调用链和当前 Python 栈，这个 gather 位于 GDN/KDA metadata 构建附近，最可能来自
`mamba_get_block_table_tensor()` / block-table gather，而不是 B300 数据面的某个随机 gather。

### 3.3 Python 与 native 调用栈

TP1/TP2/TP6 的 Python 主线程都停在：

```text
DLCModelRunner._dummy_run
  -> _build_attention_metadata
  -> _build_attn_group_metadata
  -> GDNAttentionMetadataBuilder.build
  -> gdn_attn.py:339
       block_table_tensor[spec_sequence_masks_cpu, : self.num_spec + 1]
```

TP1 的 locals 显示当前 dummy batch：

```text
num_tokens = 16
num_reqs = 8
num_scheduled_tokens_list = [2, 2, 2, 2, 2, 2, 2, 2]
num_spec_decodes = 8
num_spec_decode_tokens = 16
profile_seq_lens = 256
capture_descs 总数 = 4
当前是第一个 profile desc
```

TP1/TP2/TP6 的 native 栈继续显示：

```text
gdn_attn.py:339
  -> index_Tensor meta / make_info
  -> handle_h2d_copy
  -> dlcStreamSynchronize
  -> DriverDevice::Wait
```

TP0/TP3/TP4/TP5/TP7 则在 `_dummy_run()` 返回前的
`torch.dlc.synchronize()` 等待，native 栈为 `synDeviceSynchronize -> DeviceSubmissionState::Wait`。

需要注意：主机侧停在 `handle_h2d_copy -> dlcStreamSynchronize`，只能说明它在等待设备流，
**不能单独证明 H2D copy 是第一个错误点**。设备已经在 gather 上 Halted/FAIL 后，后续任何
H2D 或 stream synchronize 都会永久等待。因果方向仍需在 gather launch 前后加轻量现场才能确认。

### 3.4 与历史故障的共同点

2026-11-16 曾遇到非常相似的现场：

- `max-model-len=66560`、chunk=512、MTP graph、`max-num-seqs=8`；
- warmup 报 `synErrorLaunchFailure`；
- 设备停在 `custom_gather_tensor` 的 `bad_vld_base`；
- Python 栈同样落在 `gdn_attn.py:339`；
- 单独构造相同 shape/dtype 的 gather 测例，eager/graph 多轮和 8 卡并发均通过；
- 加入同步日志或改变 launch 时序后问题消失。

这些事实强烈指向 graph capture 期间的时序/竞态，而不是一个固定的 gather 形状错误。但
“异步 H2D 尚未完成，算子消费异常数据”目前仍是高优先级假设，还没有抓到 copy source/GPU
metadata 与 kernel 实际读值之间的完整证据链。

### 3.5 O0/O3 观察

当前已知的观察是：

- `DLC_SYN_COPY_ASYNC=O0` 在一部分启动场景中降低了卡死概率；
- 不设置该变量时曾概率性卡住；
- 本次现场显式设置为 `O3`，仍然卡住。

因此 **O3 不能被当作稳定规避方案**。同时，O0 也只改善部分现象，不能修复
`num_spec=2 + graph` 的 qsl 错误。

另一个必须核实的点是变量传播：API 父进程显式有 `DLC_SYN_COPY_ASYNC=O3`，但 worker 的
初始 `/proc/<pid>/environ` 中未看到该变量。由于未设置时 PyTorch 默认值也是 O3，本次现场
仍等价于 O3；但后续做 O0/O2/O3 A/B 时，必须从 worker 内部记录实际 `async_copy_level()`，
否则不同 run 的对照可能并不可信。

### 3.6 复现命令

该问题为概率性现象，单次启动成功不能证明问题消失。复测时应先停止全部 vLLM 进程并
执行 full soft reset，再在相同硬件初始状态下重复多轮。

环境变量：

```bash
export VLLM_PLUGINS=dlc,dlc_patches
export DLC_SYN_GRAPH_CACHE=1
export DLC_SYN_COPY_ASYNC=O3
export DLC_MTP_DRAFTER_CUDAGRAPH=1
export DLC_MTP_PROPOSER_CUDAGRAPH=1
export VLLM_USE_DLC_COL_MAJOR_MATMUL=1
export DLC_KPOOL_SLOT_MAPPING_CHECK=1
export PYTORCH_DLC_ALLOC_CONF=expandable_segments:True
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_LOGGING_LEVEL=INFO
```

启动命令：

```bash
vllm serve /data/models/GLM-5.3-Flash-Mix \
  --served-model-name GLM-5.3-Flash-Mix \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 128000 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --port 18005 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kernel-config '{"enable_jit_warmup": false}'
```

预期现象：服务在 graph memory profiling/warmup 阶段卡住，部分 rank 可能出现
`custom_gather_tensor` 的 `bad_vld_base: Vld base out of range`。失败 rank 集合不固定，
已观察到 `1/2/6` 和 `1/2/3` 两种组合。

---

## 4. 两个问题的关系

| 项目 | 问题一：num_spec>1 qsl | 问题二：启动卡死/bad_vld_base |
|---|---|---|
| 触发 MTP | `num_spec=2` | 当前为 `num_spec=1` |
| 触发阶段 | graph capture 中的 KPool metadata build | graph memory profiling/warmup |
| 直接报错 | `query_start_loc[0] must be 0, got 16` | `custom_gather_tensor bad_vld_base` |
| 是否稳定复现 | 是，`num_spec=2 + graph` 可稳定复现 | 概率性，受 reset/时序/日志影响 |
| O0 是否修复 | 否 | 部分场景降低概率，但不是完整修复 |
| 当前判断 | metadata buffer 生命周期 / 错误 qsl | graph capture 时序/竞态；H2D 是待验证假设 |

共同设计风险是：graph capture 需要静态、持久、生命周期明确的 metadata 和 staging buffer；
当前部分 CPU metadata 仍是临时 tensor 或共享 buffer，进入 graph 后存在被覆盖或读取旧值
的可能。

但问题一已经有明确的错误参数和校验点；问题二目前仍缺少“坏地址由哪次 copy/哪个 producer
写入”的直接证据，必须分开定位。

---

## 5. 建议的下一步

1. **先保留本次现场，确认不再需要后停止进程并执行 full soft reset，再开始下一轮实验。**
2. 在 worker 初始化日志中打印实际 `async_copy_level()`，确认 O0/O2/O3 是否真的传入
   EngineCore 和 TP worker，而不是只存在于 API 父进程。
3. 以相同命令、相同硬件 reset 状态，对 `O0/O2/O3` 各做多轮启动统计，记录成功率和首个失败点；
   每轮只改变一个变量。
4. 在 `mamba_get_block_table_tensor()` 和 `gdn_attn.py:339` 前后记录
   block table / index tensor 的 `data_ptr`、shape、stride、dtype 和设备地址区间；
   不打印大 tensor 内容，避免改变时序过多。
5. 为 graph bucket 预置持久化的 qsl、seq-lens、block-table staging buffer，避免 capture
   期间继续使用临时 pinned tensor；这与问题一的修复方向一致。
6. 若仍可在 gather 前加入极轻量 stream/event 断点，验证 gather launch 前所有 producer
   H2D 是否已完成；不要直接使用会改变 launch 顺序的大量日志作为生产修复。
7. 保留 `DLC_KPOOL_SLOT_MAPPING_CHECK=1`。关闭校验只会隐藏错误 qsl，已经证实会产生错误输出。

---

## 6. 原始现场证据

证据目录：

```text
/root/workspace/log/hang_async_h2d_20261120/
```

关键文件：

| 文件 | 内容 |
|---|---|
| `peek_status.txt` | 8 卡状态、错误汇总 |
| `peek_stuck_0.txt` ... `peek_stuck_7.txt` | 每张卡的 PC、kernel、参数、PCIe/HBM 状态 |
| `pyspy_854799.txt` | API 等待 EngineCore 启动 |
| `pyspy_855264.txt` | EngineCore 等待 `determine_available_memory` |
| `pyspy_855709.txt` ... `pyspy_855716.txt` | 8 个 TP worker 的 Python 栈 |
| `pyspy_855709_locals.txt` | TP0 dummy run 局部变量 |
| `pyspy_855710_locals.txt` | TP1 GDN metadata/dummy batch 局部变量 |
| `pyspy_855709_native.txt` | TP0 `synDeviceSynchronize` native 栈 |
| `pyspy_855710_native.txt` | TP1 `handle_h2d_copy -> dlcStreamSynchronize` native 栈 |
| `pyspy_855711_native.txt`, `pyspy_855715_native.txt` | TP2/TP6 同 TP1 |
| `process_threads.txt` | 进程/线程状态与 wchan 快照 |
| `session_manifest.txt` | revision、命令、环境变量、进程信息 |
| `dmesg.txt` | 当前 dmesg；本轮运行后没有新的 driver error，只有此前 reset 记录 |


---

## 7. 复现验证结果（2026-11-20）

### 7.1 问题一：qsl 错误复现成功

执行步骤：

1. 停止首次卡住的服务，执行 `cltechpd_clnt -s --full`。
2. 确认 8 卡为 `Halted/PASS`。
3. 使用第 2.4 节命令，`num_speculative_tokens=2`、`DLC_SYN_COPY_ASYNC=O0` 启动。

结果：**复现成功**。服务在 graph capture 阶段退出，错误为：

```text
RuntimeError: /root/workspace/pytorch/aten/src/ATen/native/dlc/FusedOps.cpp:7847(kpool_slot_mapping_check_metadata)
kpool_slot_mapping: query_start_loc[0] must be 0, got 8
```

本配置下 TP0、TP6、TP7 均记录到了同一错误；EngineCore 随后报：

```text
RuntimeError: Worker failed ... query_start_loc[0] must be 0, got 8
RuntimeError: Engine core initialization failed.
```

关键日志：

```text
/root/workspace/log/repro_async_h2d_20261120/issue1_numspec2_qsl_run1.log
```

结论：问题一可稳定复现；错误值与 batch/graph bucket 相关，`max-num-seqs=8` 时为 `8`，
不是固定为 `16`。

### 7.2 问题二：O3 启动卡死复现成功

执行步骤：

1. 问题一退出后再次执行 `cltechpd_clnt -s --full`。
2. 确认设备复位完成。
3. 使用第 3.6 节命令，`num_speculative_tokens=1`、`DLC_SYN_COPY_ASYNC=O3` 启动。

结果：**首轮复测即复现卡死**。所有 worker 到达：

```text
Profiling CUDA graph memory: FULL=4 (largest=16)
```

随后超过 60 秒无推进，EngineCore 报：

```text
No available shared memory broadcast block found in 60 seconds.
This typically happens when some processes are hanging or doing some time-consuming work.
```

本次硬件现场：

- API `911413`、EngineCore `911942`、TP0-TP7 worker `912373`-`912380` 仍保持运行；
- `cltech1/cltech2/cltech3`：`Halted + FAIL`，均停在 `custom_gather_tensor`；
- 其余卡：`Running/PASS`，停在 allreduce/halting；
- 三张失败卡均为 `bad_vld_base: Vld base out of range`；
- TP1/TP2/TP3 的 Python 栈停在 `gdn_attn.py:339`；
- TP1 native 栈为 `handle_h2d_copy -> dlcStreamSynchronize -> DriverDevice::Wait`；
- TP0/TP4/TP5/TP6/TP7 在 `_dummy_run()` 的 `torch.dlc.synchronize()` 等待。

与首次现场的 `1/2/6` 相比，本次失败 rank 为 `1/2/3`。失败 rank 集合会变化，进一步支持
该问题具有时序/竞态特征，而不是固定 rank 或固定输入 shape 的确定性问题。

关键日志和现场：

```text
/root/workspace/log/repro_async_h2d_20261120/issue2_mtp1_o3_run1.log
/root/workspace/log/repro_async_h2d_20261120/issue2_run1_peek_status.txt
/root/workspace/log/repro_async_h2d_20261120/issue2_run1_peek_stuck_1.txt
/root/workspace/log/repro_async_h2d_20261120/issue2_run1_peek_stuck_2.txt
/root/workspace/log/repro_async_h2d_20261120/issue2_run1_peek_stuck_3.txt
/root/workspace/log/repro_async_h2d_20261120/issue2_run1_pyspy_912374_native.txt
```

结论：`DLC_SYN_COPY_ASYNC=O3` 不能稳定规避启动卡死；该问题在当前版本和当前启动配置下
仍可复现。

采集命令：

```bash
/usr/local/chipltech/cl_tools/CsrAccess/peek.sh -s status
/usr/local/chipltech/cl_tools/CsrAccess/peek_stuck.sh -t N
py-spy dump --pid <pid> --full-filenames
py-spy dump --pid <pid> --locals --full-filenames
py-spy dump --pid <pid> --native --full-filenames
```

---

## 8. 2026-11-20：问题一进一步定位

在问题一稳定复现后，对 `vllm-cl` 做了临时调试，分别记录：

- `common_attn_metadata.query_start_loc` 的 GPU shape、storage offset、data_ptr；
- `common_attn_metadata.query_start_loc_cpu` 的 CPU 真值；
- `compute_kpool_slot_mappings_dlc()` 内 clone 后的 `starts` 指针、offset 和 CPU 值；
- 当前是否处于 `torch.dlc.is_current_stream_capturing()`。

调试代码只用于本地实验，已撤回，不在仓库中保留。

### 8.1 capture 前 metadata 是正确的

在 `num_speculative_tokens=2`、`max-num-seqs=8` 的 graph memory profiling 阶段，非 capture 调用
观察到两组 qsl：

```text
target/verify: num_tokens=24, qsl_cpu=[0,3,6,9,12,15,18,21,24]
draft:         num_tokens=8,  qsl_cpu=[0,1,2,3,4,5,6,7,8]
```

两组 GPU qsl 的 `shape=(9,)`、`storage_offset=0`、`data_ptr` 也正常；`starts` clone 后
CPU 值与 GPU 源 qsl 的 CPU 真值一致。因此问题不是：

- Python 侧 qsl 计算错误；
- qsl 的 shape/stride 错误；
- 静态 `self.arange` 构造错误；
- KPool 的 `num_reqs` 或 padding 公式错误。

### 8.2 第一次分叉发生在 capture 内

真正失败的是 `capturing=True` 的 draft KPool 调用。此时：

- qsl 指针、shape、storage offset 仍然正常；
- `starts` 的指针和 offset 也正常；
- C++ `kpool_slot_mapping_check_metadata()` 对 qsl 做 host D2H 后读到首元素 `8`；
- 该调用其余 rank 全部同时失败，说明不是单个 rank 的输入数值差异。

需要注意，`pytorch/aten/src/ATen/native/dlc/FusedOps.cpp:7818-7820` 已经写明：

> host-synchronizing metadata checks 默认启用；在 captured graph 内应设置
> `DLC_KPOOL_SLOT_MAPPING_CHECK=0`，因为 D2H copy 会破坏 capture。

因此 **capture 内执行这个 host check 本身就是不受支持的路径**，它会在 capture 期间产生
不可靠的 host 读数或直接失败。首个 `got 8` 错误至少部分是该 check 在 capture 内的伪/异常读数；
但下面证明不能只靠关闭 check 解决。

### 8.3 `CHECK=0` 能让服务启动，但输出仍错误

保持 `num_spec=2 + graph + O0`，只设置：

```bash
export DLC_KPOOL_SLOT_MAPPING_CHECK=0
```

服务可以完成 capture 并进入 `Application startup complete`。但用历史 eager 正确性测例的同一 prompt：

```text
prompt=[100,101,102,103,104,105,106,107], max_tokens=32, temperature=0
```

当前 graph 输出为：

```text
[198, 7, 21, 12, 21, 12, 21, 12, ...]
```

历史 eager `num_spec=2` 正确输出为：

```text
[198, 147, 100, 36478, 100, 36478, ...]
```

首 token 后即分叉。**进一步对照显示 eager、target graph、drafter/proposer graph 的不同
组合都得到同样错误序列**；使用 `USE_CPU_FOR_COLLECT_OP=True` + prefix cache 关闭继续对照后，
输出恢复为历史正确序列 `[198,147,100,36478,...]`。因此该输出分叉属于 MTP=2 + normal
DLCCL/配置路径的独立问题，不能单独归因于 graph qsl。关闭 check 只绕过了 capture 内的
host 校验，不能作为最终修复。

日志和响应：

```text
/root/workspace/log/repro_async_h2d_20261120/issue1_numspec2_qsl_checkoff.log
/root/workspace/log/repro_async_h2d_20261120/issue1_numspec2_checkoff_8tok_tokens.json
```

### 8.4 `starts` 不 clone 也不能修复

为排除 KPool 内临时 `starts` clone 生命周期，临时增加 `DLC_KPOOL_NO_CLONE=1`：

```text
starts = query_start_loc[: num_reqs + 1].to(torch.int32)
```

结果不是启动成功，而是错误前移为：

```text
kpool_slot_mapping: sum(seq_lens) must equal num_tokens (8), got 29
```

说明 capture 内 KPool 看到的边界缓冲发生了异常；但由于 eager + normal DLCCL 也能得到同样的
错误输出，不能仅据此认定 graph qsl 是独立根因。clone 问题仍需与 MTP=2/normal DLCCL
的输出分叉分开验证。

另一个对照是使用按 `(device, num_reqs, num_tokens)` 持久化的 `starts` buffer，并在
`DLC_KPOOL_SLOT_MAPPING_CHECK=0` 下复用该 buffer，而不是每次 clone。服务同样可以启动，
但固定 8-token prompt 输出仍为：

```text
[198, 7, 21, 12, 21, 12, ...]
```

因此 KPool 内部的 `starts` 临时 buffer 不是当前固定错误输出的单独主因；进一步验证应同时
覆盖 MTP=2 normal DLCCL 与确定性通信两条路径。

### 8.5 当前定位结论

目前证据支持以下结论：

1. **非 capture 路径 qsl 正确**；错误只在 `capturing=True` 的 draft KPool 路径出现。
2. capture 内的 host-side D2H metadata check 本身不受支持，`CHECK=1` 会先于实际 model 输出暴露
   错误，甚至在部分情况下给出不可靠读数。
3. 关闭 check 后实际输出仍错误，但 eager + normal DLCCL 也错误；该输出问题不能单独作为
   graph qsl/staging 出错的证据。
4. 单纯去掉 `starts.clone()` 不能修复；但当前输出错误与通信/MTP=2 路径混杂，尚不能把
   KPool 局部 clone 定为独立根因。
5. `num_spec=1` graph path 已验证；`num_spec=2` 的 graph qsl 是否存在独立内容错误，仍需在
   确定性通信基线下复测。

**进一步对照（2026-11-20）**：

| 配置 | 8-token 输出 |
|---|---|
| `num_spec=2`, target graph + drafter graph + proposer graph, CHECK=0 | `[198, 7, 21, 12, ...]` |
| `num_spec=2`, target graph, proposer graph off, CHECK=0 | `[198, 7, 21, 12, ...]` |
| `num_spec=2`, target graph, drafter/proposer graph off, CHECK=0 | `[198, 7, 21, 12, ...]` |
| `num_spec=2`, `--enforce-eager`, CHECK=1 | `[198, 7, 21, 12, ...]` |
| `num_spec=2`, eager + CPU collective + prefix cache off, CHECK=1 | `[198,147,100,36478,...]` |
| `num_spec=2`, graph + native DLCCL + ordered BF16 allreduce, prefix cache off, CHECK=0 | `[271,98886,271,98886,...]` |
| `num_spec=2`, eager + native DLCCL + ordered BF16 allreduce, prefix cache off, CHECK=1 | `[271,98886,271,98886,...]` |

因此当前固定的错误输出在 **所有 MTP graph 组合和 eager 下都存在**，不是由某一个 graph
开关单独引入；在确定性 CPU communication + prefix cache 关闭下输出恢复正确。该输出问题
应归入 MTP=2 + normal DLCCL/配置路径，而不是 graph qsl。

补充：`DLCCL_ORDERED_BF16_ALLREDUCE=1` 在 graph 和 eager 下都没有恢复正确输出，说明
该开关在当前 8-TPU/算法选择路径下不足以形成可用于正确性判断的确定性基线；需要确认它是否
命中了实际使用的 allreduce kernel algorithm。

### 8.5.1 具体涉及的代码

当前最需要检查和修改的代码是：

1. `vllm/vllm/v1/spec_decode/llm_base_proposer.py:676-681`
   - `num_speculative_tokens > 1` 时在 `propose()` 内直接改写共享
     `common_attn_metadata.query_start_loc` 和 `query_start_loc_cpu`。
   - 该 metadata 对象随后被 MTP proposer graph 使用，属于 in-place mutation。
2. `vllm-cl/vllm_cl/patch/patch_eagle.py:411-439`
   - 在 capture 外构造 `_dlc_mtp_first_pass_metadata`，但实际把 `common_attn_metadata`
     作为 positional argument 传进 `CUDAGraphWrapper`。
   - 没有为 target/draft 的 qsl 建立独立的 persistent graph input。
3. `vllm-cl/vllm_cl/patch/patch_eagle.py:462-479`
   - draft_index=0 直接返回缓存 metadata；draft_index>0 重新构建，
     但共享的 `common_attn_metadata` 已被第一段流程改写。
4. `vllm-cl/vllm_cl/compilation/dlc_graph.py:59-65`
   - 文件自身明确声明 wrapper **不保存 persistent buffers**；
   - `:180-182` 只 clone kwargs 中的 Tensor，positional 参数和
     `CommonAttentionMetadata` 内部的 qsl 不在保护范围内。
5. `vllm-cl/vllm_cl/kpool_worker_slots.py:105-131`
   - KPool 每次生成临时 `starts`，在 graph capture 内进一步放大了 metadata 生命周期问题。
6. `pytorch/aten/src/ATen/native/dlc/FusedOps.cpp:7818-7826, 7990-7992`
   - host-side metadata check 在 capture 内执行 D2H；这是当前 `got 8`/`got 29` 的直接触发点，
     但只是检测器，不是完整根因。

### 8.7 本轮状态

- 问题一已在 `num_spec=2 + graph + O0` 下复现。
- `CHECK=0` 能启动，但固定 8-token prompt 输出仍错误；eager 和所有 MTP graph 开关组合
  也得到同样错误序列，说明该输出分叉是独立的 MTP=2/normal DLCCL 问题。
- 临时移除 `starts.clone()`、或使用持久化 `starts` buffer，都不能修复实际输出。
- 当前判断：`got 8`/`got 29` 的直接代码问题是 capture 内 host-side D2H check；
  MTP proposer graph、CommonAttentionMetadata、KPool starts 是 capture metadata 生命周期的
  候选风险，但当前固定错误序列已证明不能单独归因于它们。
- 实验代码和调试打印已撤回；设备已 full reset，8 卡均为 `Halted/PASS`。
- 下一步优先对比 `_fused_mapping()` 收到的 `query_start_loc` 参数与闭包
  `common_attn_metadata.query_start_loc` 是否为同一 buffer，并尝试让 KPool 使用调用点参数。

### 8.8 排查口径纠偏（2026-11-20）

前文中的 CPU collective、`DLCCL_ORDERED_BF16_ALLREDUCE` 和不同 graph 开关的输出对照，
只用于判断“输出分叉是否由 graph qsl 引入”，不是 qsl 问题的根因路线。它们已经表明
MTP=2 的 eager 输出还会受 normal DLCCL/配置路径影响，因此**停止用 allreduce 确定性来
构建 qsl 的验收基线**。

qsl 问题应直接在 graph capture metadata 层研究：

1. `DLC_KPOOL_SLOT_MAPPING_CHECK` 保持关闭，避免 D2H host check 干扰 capture；
2. 在 capture 期间直接记录 `starts` 的实际值与来源，而不是通过最终文本输出反推；
3. 对比 `get_compressed_slot_mapping()` 的实参 qsl、闭包 `common_attn_metadata.query_start_loc`、
   `compute_kpool_slot_mappings_dlc()` 内 clone 后的 `starts`；
4. allreduce/CPU collective 的差异另行归档，仅在需要解释模型输出时再处理。

### 8.9 capture-time qsl 实测（关键证据）

在 `DLC_KPOOL_SLOT_MAPPING_CHECK=0` 下临时记录 capture 内的值，得到：

```text
num_reqs=8 num_tokens=8
q_values      = [0, 1, 2, 3, 4, 5, 6, 7, 0]
starts_values = [8, 24, 24, 24, 24, 24, 24, 24, 24]
```

其他 graph bucket 也表现为类似模式：

```text
num_reqs=6 num_tokens=6
q_values      = [0,1,2,3,4,5,0]
starts_values = [6,18,18,18,18,18,18]
```

结论非常明确：

1. capture 时 qsl 的前半部分基本是当前 draft qsl（`0,1,2,...`）；
2. `compute_kpool_slot_mappings_dlc()` 内部的临时 `starts` clone 在 capture 时读到的却是
   graph pool 的旧数据（例如 `[8,24,24,...]`）；
3. KPool kernel 实际拿到的 `starts` 就是这份错误数据；
4. `kpool_slot_mapping_check_metadata()` 的 D2H host check 只是最早发现该问题的代码。

因此，问题一的直接代码位置可以锁定为：

```text
vllm-cl/vllm_cl/kpool_worker_slots.py:105-115
```

当前代码在 capture 内执行：

```python
starts = query_start_loc[: num_reqs + 1].to(torch.int32).clone().contiguous()
```

以及后续 `starts[-1].add_(padding)`。这些操作生成的临时 staging 在 capture 内不是当前
qsl 的副本，而是旧 graph pool 内存。

此前“持久化 starts buffer”实验仍然在 capture 内执行了 `starts.copy_(qsl)`，因此没有解决
问题；下一次修复必须在非 capture 阶段把 qsl 写入 persistent buffer，capture 内只复用该
buffer，不再 clone 或 copy。若某次调用确实存在 padding，也必须在 capture 前完成 padding。

该证据也说明前文所有 CPU collective/allreduce 输出对照与 qsl 根因无关，后续不再用它们判断
qsl 问题。

### 8.10 算子入口内部实测（2026-11-20）

为排除 Python 侧代理观测，在 `pytorch/aten/src/ATen/native/dlc/FusedOps.cpp` 的
`dlc_kpool_slot_mapping_impl()` 入口临时增加 `DLC_KPOOL_SLOT_MAPPING_DEBUG` 打印，
直接输出算子实际收到的 qsl/seq_lens。关键结果：

**capture 前**：

```text
[DLC_KPOOL_OP_DEBUG] call=1 R=8 T=8
qsl=[0,1,2,3,4,5,6,7,8]
seq_lens=[1,1,1,1,1,1,1,1]
```

**进入 graph capture 后，同一个 T=8 调用**：

```text
[DLC_KPOOL_OP_DEBUG] call=2 R=8 T=8
qsl=[8,24,24,24,24,24,24,24,24]
seq_lens=[8,24,24,24,24,24,24,24]
```

这是算子入口的真实参数，不是 Python clone 的间接推断。结论：

- `query_start_loc` 在 capture 前正确；
- 进入 capture 后，KPool 算子实际收到的是 `[8,24,24,...]`；
- `kpool_slot_mapping_check_metadata()` 的 `got 8` 不是误报，它准确检测到了这个错误输入；
- 问题锁定在 `compute_kpool_slot_mappings_dlc()` 的 capture-time staging，以及它上游传入的
  `common_attn_metadata.query_start_loc` 生命周期，而不是 KPool kernel 数学。

临时 C++ debug 代码将在记录后撤回并恢复算子库。

### 8.6 下一步实验

1. 在 capture 内自动跳过 host-side metadata check（例如根据
   `currentStreamCaptureStatus() != None`），但仅把它作为调试/避免伪失败，不作为问题一修复。
2. 为 target/verify 和每个 draft step 分别预置持久 qsl、seq_lens、`starts` staging buffer，
   在 capture 前完成初始化，capture 内只做受控 copy，不复用临时 buffer。
3. 检查 `CommonAttentionMetadata` 构造处是否能把 qsl 直接绑定到 per-graph-bucket 的
   persistent buffer，而不是在 `build_for_drafting()` 内生成/拷贝。
4. 做 `DLC_SYN_GRAPH_CACHE=0` 对照，确认 graph replay cache 是否放大了该问题。
5. 修复后重新验证：
   - `num_spec=2 + graph` 启动；
   - 8/64-token greedy 与 no-MTP baseline 完全一致；
   - 1/8 路并发和 graph/eager 对照；
   - 再推进 `num_spec=3/4/8`。

