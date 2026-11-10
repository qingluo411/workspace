# GLM-5.3-Flash DLC 适配交接

本目录保存 GLM-5.3-Flash 在 vLLM + vLLM-CL 初期适配的研究、阻断点、配置、权重映射审计和启动脚本。

- `GLM-5.3-Flash_vLLM-CL_阻断点2-7.md`：阻断点 2–7 的逐项状态、代码位置、测试边界和待验证项。
- `GLM-5.3-Flash_vLLM-CL_联调问题与修复记录.md`：真机联调（启动脚本 → KV cache 分配）中暴露的问题、根因、修复与未决项。
- `GLM-5.3-Flash_NaN问题排查总结与SOP_2026-10-22.md`：GLM-5.3-Flash 全 `!` 输出 / NaN 故障的完整定位过程；记录 KDA state pool 16 GiB 地址溢出与 sparse MLA null block 污染两个独立根因、分阶段修复验证及通用 NaN 排查 SOP。
- `KPool_09_select_topk_性能量化_2026-10-22.md`：`#09 dlc_vllm_kpool_select_topk` 与 native 回退路径的单层隔离基准；给出 prefill 约 180 token、单请求 decode 约 448 token、32 请求 decode 约 800 token/请求的交叉点、4K prefill / 长上下文 decode 的量化差距，以及最终拆除 decode-only 门禁、prefill/decode 统一走 `#07 + #09` 的决策与代价。
- `KPool_select_topk_根因分析与优化方案_2026-11-02.md`：KPool select/topk kernel 的长上下文根因、token-major 重扫/串行 head/全量排序的量化证据，以及第一轮 `head unroll + scale folding + block-table hoist` 实现和 7.1%–7.5% 实测收益。
- `GLM-5.3-Flash_vLLM-CL_适配研究.md`：模型结构与三仓库执行路径研究。
- `op_requests/`：融合算子需求 issue 草稿（18 个，按 `op_request.yml` 模板字段编写；已按覆盖核查分成
  「完全新算子 7 / 需扩展 3 / 需修复 1 / 可复用撤回 7」四类，实际提交 11 条，每份文件内含「需求分类」段；
  分类于 2026-09-22 三轮修正：#02（null block 语义）与 #18（页几何全部 512B 对齐）由提交改为撤回，
  #16（clamp 激活）由「扩展 silu_and_mul」改为「新增独立算子」，#13 复核后维持「扩展」；
  **2026-09-28：真机跑通后 #05 由「可复用撤回」重新提为 P0 新增算子**，见 `给算子组_05_state_pool_index_perf.md`）。
- `op_requests/接入清单_2026-09-28.md`：算子接入清单（已接入 5 条 / 待接线 2 条 / 等算子组 11 条）。
- `给算子组_05_state_pool_index_perf.md`：2026-09-28 真机 profile 证据 —— KDA 状态池在 Mamba 页内**非连续 view** 下，通用 `index_select`/`index_copy_` 掉进 host 逐元素路径，占 prefill 单步 64%；#05 因此由「可复用撤回」重新提为 P0。
- `op_requests/00_已有算子覆盖核查.md`：18 条需求是否已被 DLC 现有算子覆盖的实测核查结论、约束与待确认问题。
- `给pytorch后端_dlc_strided_copy_bug.md`：pytorch DLC 后端 strided→contiguous 拷贝缺陷（大张量 + 非 512B 对齐基址下静默算错）的 bug issue 草稿，含最小复现与影响面评估。
- `性能优化点清单_2026-09-29.md`：P0–P2 优化点、逐条归因数据、覆盖判定与建议执行顺序。
- `元数据H2D归因与合并方案_2026-10-09.md`：每步 host→device metadata 拷贝的代码位置归因、
  单次 44 µs 固定成本的实测，以及三级合并方案（已跳过 indexer prefill chunks / 两段合一 / CpuGpuBuffer 打包）。
- `_fused_operand_方案_2026-10-09.md`：`patch_kda.py` 里 `_fused_operand`（decode 每步 306 次 view 物化）
  的根因（`KdaRecurrentDecode.cpp` 的 contiguous/offset/512B 行校验）、9 个操作数逐个定源、
  A~F 消解路径与给算子组的 ABI 扩展 prompt。
- `kpool_mla_dataflow.md`：KPool MLA 选池链路（S0–S11）的理论数据流与算子覆盖对照表。
- `拆池_mamba_state_pool.md`：Mamba/KDA 状态拆池方案与页表实现说明。
- `集合通信同步归因_2026-10-10.md`：用 `DLCCL_SYNC_ALL_MACHINE=1` 把集合通信拆成「等同步」与
  「ring 本体」，结论是 decode 步约 70% 的集合通信时间是等待、ring 仅 0.55 ms/次，且全程无计算/通信重叠。
  文档后半段进一步定位到「等在等什么」：屏障放行跨卡只差 0.04 ms、各卡有效设备时间 94.3–95.9 ms（±0.8%），
  skew 全部来自 host 侧（3888 次下发 / 379 ms 空洞 / 每步 11 次 `rows.tolist()` 阻塞读回）。
  原始数据与复现脚本在 `measurements/2026-10-10_sync_split/`。
- `下一步工作计划_2026-10-10.md`：把「理想算子流水线」的差距落成可验收工作项，分四组
  （A 算子组 7 项 / B 插件侧 6 项 / C 框架侧 4 项 / D 图捕获与并行 4 项），
  含 M1–M8 量化判据（每步 H2D→1 次、D2H→1 次、下发次数→≤800、步长目标 150–200 ms 量级）与执行顺序。
- `measurements/e2e_baseline_2026-10-10.md`：**端到端性能当前基线**（#05 融合算子接入后，无 profiler）——
  1223 token prompt 的 TTFT ≈ 4.7 s、单请求 decode **≈233 ms/token（4.4 tok/s）**、3 并发聚合 8.2 tok/s；
  后续端到端改动一律与此对照。复现脚本 `probes/bench_e2e_glm53.py`。
- `op_requests/probes/`：覆盖核查用的 DLC 实测脚本（自带 reference 与误差打印，可复现核查数据）。
- `audit_glm5_weight_mapping.py` / `GLM-5.3-Flash_weight_mapping_audit.json`：不加载 tensor payload 的权重键和 header 审计。
- `start_glm5_flash_vllm_cl.sh`：初期服务启动模板，显式带 `--block-size 256` 和 `--enforce-eager`。

### 启动脚本的 profiling 开关

三种互不影响的采集方式，都由脚本设置环境变量后照常起服务：

| 开关 | 采集内容 | 产物 | 解析 |
|---|---|---|---|
| （默认） | 无 | — | — |
| `--profile[=<dir>]` | vLLM torch profiler（host 侧算子耗时/次数）+ DLC cycle 计数；另写 runtime per-thread trace（含设备 cycles） | `<dir>/*.pt.trace.json.gz`、`<dir>/syn_trace/syn_trace_<pid>.json` | `/tmp/synprof/analyze_worker.py`；cyles 占比见 `measurements/2026-10-09_cycles/op_share_cycles.py` |
| `--kernel-log[=<dir>]` | Synapse 每次下发的 kernel 明细日志：`DLCTensor[...] shape/stride/dtype/offset`、`DLCScalar[...]`，以及 `cycles/ops/bytes/crt` | `<dir>/<prefix>_<pid>.ansi`（默认 `dir=/root/workspace/log/syn_log`） | `python /root/workspace/llama2-fine-tune/tool.py <dir>` → 同目录 `*_kernels.txt` |

`--kernel-log` 等价于手工设置：

```bash
export DLC_SYN_LOG_DIR=/root/workspace/log/syn_log   # 可被 --kernel-log=<dir> / GLM5_SYN_LOG_DIR 覆盖
export DLC_SYN_VERBOSE=1                            # GLM5_SYN_VERBOSE 覆盖；1=下发名，4=额外 kernel args
export DLC_SYN_DEBUG=1
export DLC_SYN_PROF_CYCLE=1
```

tensor 形状来自 `DLCSynapse/synapse_core/synapse_core.cpp:2204` 的 `DEBUG && PROF_CYCLE` 分支
（`kernelArgsToString`，`external_includes/kernel_tensor.hpp:195`），与 `VERBOSE` 无关；
`VERBOSE` 只控制下发名等按级别的输出。每次下发一条记录、每 rank 一个文件，日志量很大，建议只跑短窗口。

### 集合通信等待归因：Sync-All-Machine

只要开了 `--profile` **或** `--kernel-log`，脚本都会额外导出 `DLCCL_SYNC_ALL_MACHINE=1`
（`GLM5_DLCCL_SYNC_ALL_MACHINE=0` 可关闭）。DLCCL 在每次集合通信前插入一个显式同步 kernel
（见 `DLC_CL/README.md:517` "Sync-All-Machine Kernel"），这样：

- `--profile` 的 trace 里能把「等其它 rank 到齐」和「真正的 RDMA 传输」分成两段，
  sync kernel 长 / 集合通信本体短 ⇒ 瓶颈是 rendezvous 延迟而非带宽；
- `--kernel-log` 里会多出一条同步 kernel 的下发记录（kernel 名 `custom_sync_all_machine`，
  见 `DLC_CL/src/enqueue.cc:2135`），可直接核对其 cycles 占比。

该特性默认关闭，会给每次集合通信加一次同步开销，只在 profiling 时打开。

本快照只声明源码、静态检查和 CPU/native reference 的阶段性结果；没有加载真实权重、启动服务或在 DLC 设备上验收。启动脚本中的默认模型和 vLLM 路径可通过环境变量覆盖。模型文件传输仍由外部 SCP 流程负责，不包含在本仓库提交中。
